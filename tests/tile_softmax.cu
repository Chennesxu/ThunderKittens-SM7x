#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "row_reduce_reference.cuh"
#include "test_utils.cuh"
#include "tk_sm7x/softmax.cuh"

namespace {

constexpr int kCells = 256;
constexpr int kOutputStride = 264;
constexpr double kBound = 512.0 * 0x1p-23;

__device__ void scalar_reference(
    tk_sm7x::st<float, 16, 16, tk_sm7x::row_major>& destination,
    const tk_sm7x::st<float, 16, 16, tk_sm7x::row_major>& source) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (lane < 16) {
        const std::size_t offset = static_cast<std::size_t>(lane) * 16;
        const float* row = source.data + offset;
        const float maximum = tk_sm7x::test::canonical_row<tk_sm7x::detail::row_max_op>(row);
        float exponent[16];
#pragma unroll
        for (int col = 0; col < 16; ++col) exponent[col] = expf(row[col] - maximum);
        const float denominator =
            tk_sm7x::test::canonical_row<tk_sm7x::detail::row_sum_op>(exponent);
#pragma unroll
        for (int col = 0; col < 16; ++col) {
            destination.data[offset + col] = exponent[col] / denominator;
        }
    }
    __syncwarp(0xffffffffu);
}

__device__ void normalize(const __half* a, int lda, int warps, float* before,
                          float* after, float* output, float* repeated, float* scalar) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const std::size_t group = static_cast<std::size_t>(blockIdx.x) * warps + warp;
    __shared__ __align__(32) tk_sm7x::st<__half, 16, 16, tk_sm7x::row_major> as[4];
    __shared__ __align__(32) tk_sm7x::st<__half, 16, 16, tk_sm7x::col_major> bs[4];
    __shared__ __align__(32) tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> scratch[4];
    __shared__ __align__(32) tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> dst[4];
    __shared__ __align__(32) tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> control[4];
    for (int i = lane; i < kCells; i += 32) {
        as[warp].data[i] = a[group * 16 * lda + static_cast<std::size_t>(i / 16) * lda + i % 16];
        bs[warp].data[i] = __float2half(i / 16 == i % 16 ? 1.0f : 0.0f);
        dst[warp].data[i] = nanf("");
    }
    __syncwarp(0xffffffffu);
    tk_sm7x::rt_a af;
    tk_sm7x::rt_b bf;
    tk_sm7x::rt_c cf;
    tk_sm7x::load(af, as[warp], 0);
    tk_sm7x::load(bf, bs[warp], 0);
    tk_sm7x::zero(cf);
    tk_sm7x::mma(cf, af, bf);
    tk_sm7x::store(scratch[warp], cf);
    __syncwarp(0xffffffffu);
    for (int i = lane; i < kCells; i += 32) before[group * kCells + i] = scratch[warp].data[i];
    __syncwarp(0xffffffffu);
    scalar_reference(control[warp], scratch[warp]);
    for (int i = lane; i < kCells; i += 32) {
        scalar[group * kOutputStride + 4 + i] = control[warp].data[i];
    }
    tk_sm7x::row_softmax(dst[warp], cf, scratch[warp]);
    for (int i = lane; i < kCells; i += 32) {
        output[group * kOutputStride + 4 + i] = dst[warp].data[i];
        scratch[warp].data[i] = nanf("");
    }
    __syncwarp(0xffffffffu);
    tk_sm7x::row_softmax(dst[warp], cf, scratch[warp]);
    for (int i = lane; i < kCells; i += 32) repeated[group * kOutputStride + 4 + i] = dst[warp].data[i];
    __syncwarp(0xffffffffu);
    tk_sm7x::store(scratch[warp], cf);
    __syncwarp(0xffffffffu);
    for (int i = lane; i < kCells; i += 32) after[group * kCells + i] = scratch[warp].data[i];
}

}  // namespace

extern "C" __global__ void tile_softmax(const __half* a, int lda, int warps,
    float* before, float* after, float* output, float* repeated, float* scalar) {
    normalize(a, lda, warps, before, after, output, repeated, scalar);
}

namespace {

float logits(int group, int row, int col, int kind) {
    if (kind == 1) return 0.0f;
    if (kind == 2) return (row % 2 == 0 ? 1000.0f : -1000.0f) +
        static_cast<float>(col + row + group) / 2.0f;
    if (kind == 3) {
        const int mantissa = 1024 + (137 * row + 79 * col + 53 * group) % 1024;
        const int exponent = (row + 2 * col + group) % 7 - 3;
        return std::ldexp(static_cast<float>(mantissa), exponent - 10);
    }
    return static_cast<float>((5 * col + 3 * row + 7 * group) % 16 - 8) / 8.0f +
        static_cast<float>(row) / 2.0f + static_cast<float>(group) / 4.0f;
}

bool run_case(const char* name, int kind, int warps, int blocks, int lda) {
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
    float* scalar = nullptr;
    bool ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(&a, host_a.size() * sizeof(__half)), "allocate A");
    for (float** p : {&before, &after}) {
        ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(p, cells * sizeof(float)), "allocate logits") && ok;
    }
    for (float** p : {&output, &repeated, &scalar}) {
        ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(p, output_cells * sizeof(float)), "allocate output") && ok;
    }
    if (ok) {
        std::memcpy(a, host_a.data(), host_a.size() * sizeof(__half));
        for (float* p : {before, after}) for (std::size_t i = 0; i < cells; ++i) p[i] = std::numeric_limits<float>::quiet_NaN();
        for (float* p : {output, repeated, scalar}) for (std::size_t i = 0; i < output_cells; ++i) p[i] = std::numeric_limits<float>::quiet_NaN();
        tile_softmax<<<blocks, warps * 32>>>(a, lda, warps, before, after, output, repeated, scalar);
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
                    const double scalar_error = std::fabs(static_cast<double>(scalar[i]) - want);
                    if (error > max_error) max_error = error;
                    if (!std::isfinite(output[i]) || output[i] < 0.0f || output[i] > 1.0f ||
                        error > kBound || repeated[i] != output[i] ||
                        !std::isfinite(scalar[i]) || scalar_error > kBound ||
                        output[i] != scalar[i]) {
                        std::fprintf(stderr, "%s softmax group=%d row=%d col=%d got=%.17g repeat=%.17g scalar=%.17g want=%.17g\n", name, g, r, c, static_cast<double>(output[i]), static_cast<double>(repeated[i]), static_cast<double>(scalar[i]), want);
                        ok = false;
                    }
                    output_sum += output[i];
                }
                if (std::fabs(output_sum - 1.0) > kBound || !std::isfinite(output_sum)) ok = false;
            }
            const float sentinel = std::numeric_limits<float>::quiet_NaN();
            for (float* p : {output, repeated, scalar}) for (int i : {0, 1, 2, 3, 260, 261, 262, 263}) {
                if (std::memcmp(&p[static_cast<std::size_t>(g) * kOutputStride + i], &sentinel, sizeof(float)) != 0) ok = false;
            }
        }
        std::printf("%s %s warps=%d blocks=%d lda=%d: %s max-error=%.17g bound=%.17g\n", "public", name, warps, blocks, lda, ok ? "PASS" : "FAIL", max_error, kBound);
    }
    for (void* p : {static_cast<void*>(a), static_cast<void*>(before), static_cast<void*>(after), static_cast<void*>(output), static_cast<void*>(repeated), static_cast<void*>(scalar)}) {
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
    for (int warps : {1, 4}) {
        ok = run_case("variable", 0, warps, 2, 24) && ok;
        ok = run_case("zeros", 1, warps, 2, 16) && ok;
        ok = run_case("large-offset", 2, warps, 2, 24) && ok;
        ok = run_case("rounded", 3, warps, 2, 24) && ok;
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
