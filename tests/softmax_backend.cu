#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "test_utils.cuh"
#include "tk_sm7x/ptx_backend.cuh"

namespace {

constexpr int kCells = 256;
constexpr int kOutputStride = 264;
constexpr double kBound = 512.0 * 0x1p-23;
using M8 = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm70>;
#if defined(KITTENS_SM75)
using M16 = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm75>;
#endif

template <class Backend>
__device__ void normalize(const __half* a, int lda, int warps, float* before,
                          float* after, float* output, float* repeated) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const std::size_t group = static_cast<std::size_t>(blockIdx.x) * warps + warp;
    __shared__ __align__(32) __half as[4][kCells];
    __shared__ __align__(32) __half bs[4][kCells];
    __shared__ __align__(32) float scratch[4][kCells];
    __shared__ __align__(32) float dst[4][kCells];
    for (int i = lane; i < kCells; i += 32) {
        as[warp][i] = a[group * 16 * lda + static_cast<std::size_t>(i / 16) * lda + i % 16];
        bs[warp][i] = __float2half(i / 16 == i % 16 ? 1.0f : 0.0f);
        dst[warp][i] = nanf("");
    }
    __syncwarp(0xffffffffu);
    typename Backend::fragment_a af;
    typename Backend::fragment_b bf;
    typename Backend::accumulator cf;
    Backend::load_a(af, as[warp], 16);
    Backend::load_b(bf, bs[warp], 16);
    Backend::clear(cf);
    Backend::mma(cf, af, bf);
    Backend::store(scratch[warp], cf, 16);
    __syncwarp(0xffffffffu);
    for (int i = lane; i < kCells; i += 32) before[group * kCells + i] = scratch[warp][i];
    __syncwarp(0xffffffffu);
    Backend::softmax(dst[warp], cf, scratch[warp]);
    for (int i = lane; i < kCells; i += 32) {
        output[group * kOutputStride + 4 + i] = dst[warp][i];
        scratch[warp][i] = nanf("");
    }
    __syncwarp(0xffffffffu);
    Backend::softmax(dst[warp], cf, scratch[warp]);
    for (int i = lane; i < kCells; i += 32) repeated[group * kOutputStride + 4 + i] = dst[warp][i];
    __syncwarp(0xffffffffu);
    Backend::store(scratch[warp], cf, 16);
    __syncwarp(0xffffffffu);
    for (int i = lane; i < kCells; i += 32) after[group * kCells + i] = scratch[warp][i];
}

}  // namespace

extern "C" __global__ void softmax_m8(const __half* a, int lda, int warps,
    float* before, float* after, float* output, float* repeated) {
    normalize<M8>(a, lda, warps, before, after, output, repeated);
}

#if defined(KITTENS_SM75)
extern "C" __global__ void softmax_m16(const __half* a, int lda, int warps,
    float* before, float* after, float* output, float* repeated) {
    normalize<M16>(a, lda, warps, before, after, output, repeated);
}
#endif

namespace {

float logits(int group, int row, int col, int kind) {
    if (kind == 1) return 0.0f;
    if (kind == 2) return (row % 2 == 0 ? 1000.0f : -1000.0f) +
        static_cast<float>(col + row + group) / 2.0f;
    return static_cast<float>((5 * col + 3 * row + 7 * group) % 16 - 8) / 8.0f +
        static_cast<float>(row) / 2.0f + static_cast<float>(group) / 4.0f;
}

bool run_case(bool m16, const char* name, int kind, int warps, int blocks, int lda) {
    const int groups = warps * blocks;
    const std::size_t cells = static_cast<std::size_t>(groups) * kCells;
    const std::size_t output_cells = static_cast<std::size_t>(groups) * kOutputStride;
    std::vector<__half> host_a(static_cast<std::size_t>(groups) * 16 * lda);
    for (int g = 0; g < groups; ++g) {
        for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
            host_a[static_cast<std::size_t>(g) * 16 * lda + static_cast<std::size_t>(r) * lda + c] =
                __float2half(logits(g, r, c, kind));
        }
    }
    __half* a = nullptr;
    float *before = nullptr, *after = nullptr, *output = nullptr, *repeated = nullptr;
    bool ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(&a, host_a.size() * sizeof(__half)), "allocate A");
    for (float** p : {&before, &after}) {
        ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(p, cells * sizeof(float)), "allocate logits") && ok;
    }
    for (float** p : {&output, &repeated}) {
        ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(p, output_cells * sizeof(float)), "allocate output") && ok;
    }
    if (ok) {
        std::memcpy(a, host_a.data(), host_a.size() * sizeof(__half));
        for (float* p : {before, after}) for (std::size_t i = 0; i < cells; ++i) p[i] = std::numeric_limits<float>::quiet_NaN();
        for (float* p : {output, repeated}) for (std::size_t i = 0; i < output_cells; ++i) p[i] = std::numeric_limits<float>::quiet_NaN();
#if defined(KITTENS_SM75)
        if (m16) softmax_m16<<<blocks, warps * 32>>>(a, lda, warps, before, after, output, repeated);
        else
#else
        static_cast<void>(m16);
#endif
        softmax_m8<<<blocks, warps * 32>>>(a, lda, warps, before, after, output, repeated);
        ok = tk_sm7x::test::cuda_ok(cudaGetLastError(), "softmax launch") && ok;
        ok = tk_sm7x::test::cuda_ok(cudaDeviceSynchronize(), "softmax synchronize") && ok;
    }
    double max_error = 0.0;
    if (ok) {
        for (int g = 0; g < groups; ++g) {
            for (int r = 0; r < 16; ++r) {
                double maximum = -std::numeric_limits<double>::infinity();
                for (int c = 0; c < 16; ++c) {
                    const std::size_t i = static_cast<std::size_t>(g) * kCells + r * 16 + c;
                    const float want = logits(g, r, c, kind);
                    if (!std::isfinite(before[i]) || before[i] != want || after[i] != want) {
                        std::fprintf(stderr, "%s logits group=%d row=%d col=%d before=%g after=%g want=%g\n", name, g, r, c, before[i], after[i], want);
                        ok = false;
                    }
                    if (want > maximum) maximum = want;
                }
                double exponent[16], sum = 0.0, output_sum = 0.0;
                for (int c = 0; c < 16; ++c) {
                    exponent[c] = std::exp(static_cast<double>(logits(g, r, c, kind)) - maximum);
                    sum += exponent[c];
                }
                for (int c = 0; c < 16; ++c) {
                    const std::size_t i = static_cast<std::size_t>(g) * kOutputStride + 4 + r * 16 + c;
                    const double want = exponent[c] / sum;
                    const double error = std::fabs(static_cast<double>(output[i]) - want);
                    if (error > max_error) max_error = error;
                    if (!std::isfinite(output[i]) || output[i] < 0.0f || output[i] > 1.0f ||
                        error > kBound || repeated[i] != output[i]) {
                        std::fprintf(stderr, "%s softmax group=%d row=%d col=%d got=%.17g repeat=%.17g want=%.17g\n", name, g, r, c, static_cast<double>(output[i]), static_cast<double>(repeated[i]), want);
                        ok = false;
                    }
                    output_sum += output[i];
                }
                if (std::fabs(output_sum - 1.0) > kBound || !std::isfinite(output_sum)) ok = false;
            }
            const float sentinel = std::numeric_limits<float>::quiet_NaN();
            for (float* p : {output, repeated}) for (int i : {0, 1, 2, 3, 260, 261, 262, 263}) {
                if (std::memcmp(&p[static_cast<std::size_t>(g) * kOutputStride + i], &sentinel, sizeof(float)) != 0) ok = false;
            }
        }
        std::printf("%s %s warps=%d blocks=%d lda=%d: %s max-error=%.17g bound=%.17g\n", m16 ? "m16" : "m8", name, warps, blocks, lda, ok ? "PASS" : "FAIL", max_error, kBound);
    }
    for (void* p : {static_cast<void*>(a), static_cast<void*>(before), static_cast<void*>(after), static_cast<void*>(output), static_cast<void*>(repeated)}) {
        if (p != nullptr) ok = tk_sm7x::test::cuda_ok(cudaFree(p), "cudaFree") && ok;
    }
    return ok;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selected = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selected != EXIT_SUCCESS) return selected;
    bool ok = true;
    for (bool m16 : {false, true}) {
#if !defined(KITTENS_SM75)
        if (m16) continue;
#endif
        for (int warps : {1, 4}) {
            ok = run_case(m16, "variable", 0, warps, 2, 24) && ok;
            ok = run_case(m16, "zeros", 1, warps, 2, 16) && ok;
            ok = run_case(m16, "large-offset", 2, warps, 2, 24) && ok;
        }
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
