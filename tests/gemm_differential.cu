#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "test_utils.cuh"
#include "tk_sm7x/gemm.cuh"
#include "tk_sm7x/mma.cuh"
#include "tk_sm7x/ptx_backend.cuh"

namespace {

constexpr std::uint32_t kSentinelBits = 0x7fc00000u;

#if defined(KITTENS_SM75)
constexpr int kOutputCount = 4;
constexpr int kExpectedPairCount = 6;
#else
constexpr int kOutputCount = 3;
constexpr int kExpectedPairCount = 3;
#endif

enum class Pattern { identity, signed_seed_a, signed_seed_b, cancellation };

struct GemmCase {
    const char* name;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    Pattern pattern;
};

template <class Backend>
__device__ __forceinline__ void differential_body(
    int k, const __half* a, int lda, const __half* b, int ldb, float* c, int ldc,
    __half* as, __half* bs, float* cs) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const std::size_t m0 = static_cast<std::size_t>(blockIdx.y) * 16;
    const std::size_t n0 = static_cast<std::size_t>(blockIdx.x) * 16;
    typename Backend::accumulator accumulator;
    Backend::clear(accumulator);
    for (std::size_t k0 = 0; k0 < static_cast<std::size_t>(k); k0 += 16) {
        for (int linear = lane; linear < 256; linear += 32) {
            const std::size_t row = m0 + static_cast<std::size_t>(linear / 16);
            const std::size_t col = static_cast<std::size_t>(linear % 16);
            as[linear] = a[row * static_cast<std::size_t>(lda) + k0 + col];
            bs[static_cast<std::size_t>(linear % 16) * 16 + linear / 16] =
                b[(k0 + static_cast<std::size_t>(linear / 16)) * static_cast<std::size_t>(ldb) +
                  n0 + static_cast<std::size_t>(linear % 16)];
        }
        __syncwarp(0xffffffffu);
        typename Backend::fragment_a af;
        typename Backend::fragment_b bf;
        Backend::load_a(af, as, 16);
        Backend::load_b(bf, bs, 16);
        Backend::mma(accumulator, af, bf);
        __syncwarp(0xffffffffu);
    }
    Backend::store(cs, accumulator, 16);
    __syncwarp(0xffffffffu);
    for (int linear = lane; linear < 256; linear += 32) {
        const std::size_t row = m0 + static_cast<std::size_t>(linear / 16);
        const std::size_t col = n0 + static_cast<std::size_t>(linear % 16);
        c[row * static_cast<std::size_t>(ldc) + col] = cs[linear];
    }
}

extern "C" __global__ void differential_wmma(
    int k, const __half* a, int lda, const __half* b, int ldb, float* c, int ldc) {
    __shared__ __align__(32) __half as[256];
    __shared__ __align__(32) __half bs[256];
    __shared__ __align__(32) float cs[256];
    using backend = tk_sm7x::detail::warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::target>;
    differential_body<backend>(k, a, lda, b, ldb, c, ldc, as, bs, cs);
}

extern "C" __global__ void differential_m8(
    int k, const __half* a, int lda, const __half* b, int ldb, float* c, int ldc) {
    __shared__ __align__(32) __half as[256];
    __shared__ __align__(32) __half bs[256];
    __shared__ __align__(32) float cs[256];
    using backend = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm70>;
    differential_body<backend>(k, a, lda, b, ldb, c, ldc, as, bs, cs);
}

#if defined(KITTENS_SM75)
extern "C" __global__ void differential_m16(
    int k, const __half* a, int lda, const __half* b, int ldb, float* c, int ldc) {
    __shared__ __align__(32) __half as[256];
    __shared__ __align__(32) __half bs[256];
    __shared__ __align__(32) float cs[256];
    using backend = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm75>;
    differential_body<backend>(k, a, lda, b, ldb, c, ldc, as, bs, cs);
}
#endif

float sentinel() {
    float value = 0.0f;
    std::memcpy(&value, &kSentinelBits, sizeof(value));
    return value;
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

__half value_from_q(int q) { return __float2half(static_cast<float>(q) / 8.0f); }

int signed_q(int row, int col, int seed) { return (row * 3 + col * 5 + seed) % 17 - 8; }

void fill_inputs(const GemmCase& test_case, std::vector<__half>* a, std::vector<__half>* b) {
    std::fill(a->begin(), a->end(), value_from_q(7));
    std::fill(b->begin(), b->end(), value_from_q(7));
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.k; ++col) {
            const int q = test_case.pattern == Pattern::cancellation
                ? (col % 2 == 0 ? 8 : -8)
                : signed_q(row, col, test_case.pattern == Pattern::signed_seed_a ? 1 : 9);
            (*a)[static_cast<std::size_t>(row) * test_case.lda + col] = value_from_q(q);
        }
    }
    for (int row = 0; row < test_case.k; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            int q = signed_q(row, col, test_case.pattern == Pattern::signed_seed_a ? 11 : 3);
            if (test_case.pattern == Pattern::identity) q = row == col ? 8 : 0;
            if (test_case.pattern == Pattern::cancellation) q = 8;
            (*b)[static_cast<std::size_t>(row) * test_case.ldb + col] = value_from_q(q);
        }
    }
}

std::vector<float> make_reference(const GemmCase& test_case,
                                  const std::vector<__half>& a, const std::vector<__half>& b) {
    std::vector<float> reference(static_cast<std::size_t>(test_case.m) * test_case.ldc, sentinel());
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            double sum = 0.0;
            for (int kk = 0; kk < test_case.k; ++kk) {
                sum += static_cast<double>(__half2float(a[static_cast<std::size_t>(row) * test_case.lda + kk])) *
                       static_cast<double>(__half2float(b[static_cast<std::size_t>(kk) * test_case.ldb + col]));
            }
            reference[static_cast<std::size_t>(row) * test_case.ldc + col] = static_cast<float>(sum);
        }
    }
    return reference;
}

bool check_output(const GemmCase& test_case, const char* name, const std::vector<float>& actual) {
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test_case.ldc + col;
            if (!std::isfinite(actual[index])) {
                std::fprintf(stderr, "%s %s unwritten/nonfinite output row=%d col=%d\n", test_case.name, name, row, col);
                return false;
            }
        }
        for (int col = test_case.n; col < test_case.ldc; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test_case.ldc + col;
            if (bits(actual[index]) != kSentinelBits) {
                std::fprintf(stderr, "%s %s modified C padding row=%d col=%d\n", test_case.name, name, row, col);
                return false;
            }
        }
    }
    return true;
}

bool check_pair(const GemmCase& test_case, const char* left_name, const std::vector<float>& left,
                const char* right_name, const std::vector<float>& right) {
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test_case.ldc + col;
            if (left[index] != right[index]) {
                std::fprintf(stderr, "%s direct backend mismatch %s/%s row=%d col=%d left=%g right=%g\n",
                             test_case.name, left_name, right_name, row, col, left[index], right[index]);
                return false;
            }
        }
    }
    return true;
}

bool check_cpu(const GemmCase& test_case, const char* name,
               const std::vector<float>& actual, const std::vector<float>& reference) {
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test_case.ldc + col;
            if (actual[index] != reference[index]) {
                std::fprintf(stderr, "%s %s CPU mismatch row=%d col=%d got=%g want=%g\n",
                             test_case.name, name, row, col, actual[index], reference[index]);
                return false;
            }
        }
    }
    return true;
}

bool run_case(const GemmCase& test_case, cudaStream_t stream) {
    std::vector<__half> a(static_cast<std::size_t>(test_case.m) * test_case.lda);
    std::vector<__half> b(static_cast<std::size_t>(test_case.k) * test_case.ldb);
    std::vector<float> outputs[kOutputCount];
    for (std::vector<float>& output : outputs) output.assign(static_cast<std::size_t>(test_case.m) * test_case.ldc, sentinel());
    fill_inputs(test_case, &a, &b);
    const std::vector<float> reference = make_reference(test_case, a, b);
    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_outputs[kOutputCount] = {};
    const auto release = [&]() {
        bool ok = true;
        if (device_a != nullptr) ok = tk_sm7x::test::cuda_ok(cudaFree(device_a), "cudaFree(A)") && ok;
        if (device_b != nullptr) ok = tk_sm7x::test::cuda_ok(cudaFree(device_b), "cudaFree(B)") && ok;
        for (float* output : device_outputs) if (output != nullptr) ok = tk_sm7x::test::cuda_ok(cudaFree(output), "cudaFree(C)") && ok;
        return ok;
    };
    const std::size_t a_bytes = a.size() * sizeof(__half);
    const std::size_t b_bytes = b.size() * sizeof(__half);
    const std::size_t c_bytes = outputs[0].size() * sizeof(float);
    bool prepared = tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_a), a_bytes), "cudaMalloc(A)") &&
                    tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_b), b_bytes), "cudaMalloc(B)");
    for (int index = 0; index < kOutputCount; ++index) prepared = tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_outputs[index]), c_bytes), "cudaMalloc(C)") && prepared;
    prepared = tk_sm7x::test::cuda_ok(cudaMemcpyAsync(device_a, a.data(), a_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(A)") && prepared;
    prepared = tk_sm7x::test::cuda_ok(cudaMemcpyAsync(device_b, b.data(), b_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(B)") && prepared;
    for (int index = 0; index < kOutputCount; ++index) prepared = tk_sm7x::test::cuda_ok(cudaMemcpyAsync(device_outputs[index], outputs[index].data(), c_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(C sentinel)") && prepared;
    if (!prepared) return release() && false;
    const dim3 grid(static_cast<unsigned int>(test_case.n / 16), static_cast<unsigned int>(test_case.m / 16));
    differential_wmma<<<grid, 32, 0, stream>>>(test_case.k, device_a, test_case.lda, device_b, test_case.ldb, device_outputs[0], test_case.ldc);
    if (!tk_sm7x::test::cuda_ok(cudaGetLastError(), "differential_wmma launch")) return release() && false;
    differential_m8<<<grid, 32, 0, stream>>>(test_case.k, device_a, test_case.lda, device_b, test_case.ldb, device_outputs[1], test_case.ldc);
    if (!tk_sm7x::test::cuda_ok(cudaGetLastError(), "differential_m8 launch")) return release() && false;
#if defined(KITTENS_SM75)
    differential_m16<<<grid, 32, 0, stream>>>(test_case.k, device_a, test_case.lda, device_b, test_case.ldb, device_outputs[2], test_case.ldc);
    if (!tk_sm7x::test::cuda_ok(cudaGetLastError(), "differential_m16 launch")) return release() && false;
#endif
    if (!tk_sm7x::test::cuda_ok(tk_sm7x::gemm_f16_f16_f32_nn(test_case.m, test_case.n, test_case.k,
            device_a, test_case.lda, device_b, test_case.ldb, device_outputs[kOutputCount - 1], test_case.ldc, stream),
            "gemm_f16_f16_f32_nn")) return release() && false;
    bool copied_back = tk_sm7x::test::cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    for (int index = 0; index < kOutputCount; ++index) copied_back = tk_sm7x::test::cuda_ok(cudaMemcpy(outputs[index].data(), device_outputs[index], c_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(C)") && copied_back;
#if defined(KITTENS_SM75)
    const char* names[kOutputCount] = {"wmma", "m8", "m16", "production"};
#else
    const char* names[kOutputCount] = {"wmma", "m8", "production"};
#endif
    bool matched = copied_back;
    for (int index = 0; index < kOutputCount; ++index) matched = check_output(test_case, names[index], outputs[index]) && matched;
    int pair_count = 0;
    for (int left = 0; left < kOutputCount; ++left) for (int right = left + 1; right < kOutputCount; ++right) {
        matched = check_pair(test_case, names[left], outputs[left], names[right], outputs[right]) && matched;
        ++pair_count;
    }
    if (pair_count != kExpectedPairCount) {
        std::fprintf(stderr, "%s expected %d direct backend pairs, got %d\n", test_case.name, kExpectedPairCount, pair_count);
        matched = false;
    }
    for (int index = 0; index < kOutputCount; ++index) matched = check_cpu(test_case, names[index], outputs[index], reference) && matched;
    if (matched) std::printf("%s: direct pairs=%d, CPU comparisons=%d PASS\n", test_case.name, pair_count, kOutputCount);
    return release() && matched;
}

}  // namespace

int main() {
    int device = -1;
    const int selection = tk_sm7x::test::select_sm75_device(&device);
    if (selection != EXIT_SUCCESS) return selection;
    cudaStream_t stream = nullptr;
    if (!tk_sm7x::test::cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) return EXIT_FAILURE;
    const GemmCase test_cases[] = {
        {"identity", 16, 16, 16, 24, 24, 20, Pattern::identity},
        {"signed-multi-cta", 32, 48, 32, 40, 56, 52, Pattern::signed_seed_a},
        {"long-k", 16, 32, 1024, 1032, 40, 36, Pattern::signed_seed_b},
        {"cancellation", 16, 16, 64, 72, 24, 20, Pattern::cancellation},
        {"production-small", 256, 384, 32, 40, 392, 388, Pattern::signed_seed_a},
        {"production-large", 1024, 1024, 32, 40, 1032, 1028, Pattern::signed_seed_b},
    };
    bool passed = true;
    for (const GemmCase& test_case : test_cases) passed = run_case(test_case, stream) && passed;
    const bool destroyed = tk_sm7x::test::cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return passed && destroyed ? EXIT_SUCCESS : EXIT_FAILURE;
}
