#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "test_utils.cuh"
#include "numerical_bounds.hpp"
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

enum class Pattern {
    identity,
    signed_seed_a,
    signed_seed_b,
    cancellation,
    rounded_positive,
    rounded_signed,
    rounded_mixed,
    rounded_mixed_zeros,
    rounded_cancellation,
    rounded_zero,
};

enum class ComparisonMode { exact, model };

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

struct ReferenceCell {
    double value;
    double sum_abs;
    double bound;
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

bool valid_reduction_k(int k) { return k > 0 && k % 16 == 0 && k <= 1024; }

bool is_exact_value(__half value) {
    const float logical = __half2float(value);
    return std::isfinite(logical) && std::fabs(logical) <= 1.0f &&
           std::trunc(8.0f * logical) == 8.0f * logical;
}

bool validate_exact_domain(const GemmCase& test_case, const std::vector<__half>& a,
                           const std::vector<__half>& b, bool report) {
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.k; ++col) {
            if (!is_exact_value(a[static_cast<std::size_t>(row) * test_case.lda + col])) {
                if (report) std::fprintf(stderr, "%s invalid exact A input row=%d col=%d\n", test_case.name, row, col);
                return false;
            }
        }
    }
    for (int row = 0; row < test_case.k; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            if (!is_exact_value(b[static_cast<std::size_t>(row) * test_case.ldb + col])) {
                if (report) std::fprintf(stderr, "%s invalid exact B input row=%d col=%d\n", test_case.name, row, col);
                return false;
            }
        }
    }
    return true;
}

bool run_exact_domain_self_checks() {
    const GemmCase test_case = {"exact-domain-self-check", 1, 1, 16, 16, 1, 1, Pattern::identity};
    std::vector<__half> a(16, value_from_q(8));
    std::vector<__half> b(16, value_from_q(-8));
    if (!valid_reduction_k(16) || valid_reduction_k(0) || valid_reduction_k(18) ||
        valid_reduction_k(2048) || !validate_exact_domain(test_case, a, b, false)) {
        std::fprintf(stderr, "exact-domain self-check rejected a valid fixture\n");
        return false;
    }
    a[0] = value_from_q(9);
    if (validate_exact_domain(test_case, a, b, false)) {
        std::fprintf(stderr, "exact-domain self-check accepted q outside [-8,8]\n");
        return false;
    }
    a[0] = __float2half(0.3f);
    if (validate_exact_domain(test_case, a, b, false)) {
        std::fprintf(stderr, "exact-domain self-check accepted a value off the q/8 lattice\n");
        return false;
    }
    a[0] = __float2half(std::numeric_limits<float>::infinity());
    if (validate_exact_domain(test_case, a, b, false)) {
        std::fprintf(stderr, "exact-domain self-check accepted a nonfinite value\n");
        return false;
    }
    a[0] = value_from_q(8);
    b[0] = value_from_q(-9);
    if (validate_exact_domain(test_case, a, b, false)) {
        std::fprintf(stderr, "exact-domain self-check accepted an invalid B value\n");
        return false;
    }
    return true;
}

std::uint32_t fingerprint(int row, int col, std::uint32_t seed) {
    std::uint32_t value = static_cast<std::uint32_t>(row) * 0x9e3779b9u ^
                          static_cast<std::uint32_t>(col) * 0x85ebca6bu ^ seed;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

__half rounded_value(std::uint32_t hash, Pattern pattern, bool negate) {
    const int exponent = (pattern == Pattern::rounded_mixed || pattern == Pattern::rounded_mixed_zeros)
        ? static_cast<int>((hash >> 10) % 7) - 3 : 0;
    float value = std::ldexp(1.0f + static_cast<float>(hash & 1023u) / 1024.0f, exponent);
    if (negate || ((pattern == Pattern::rounded_signed || pattern == Pattern::rounded_mixed ||
                    pattern == Pattern::rounded_mixed_zeros) && (hash >> 31))) {
        value = -value;
    }
    return __float2half(value);
}

bool is_model_pattern(Pattern pattern) {
    return pattern == Pattern::rounded_positive || pattern == Pattern::rounded_signed ||
           pattern == Pattern::rounded_mixed || pattern == Pattern::rounded_mixed_zeros ||
           pattern == Pattern::rounded_cancellation ||
           pattern == Pattern::rounded_zero;
}

bool is_model_value(__half value) {
    std::uint16_t encoded = 0;
    std::memcpy(&encoded, &value, sizeof(encoded));
    const std::uint16_t magnitude = encoded & 0x7fffu;
    if (magnitude == 0) return true;
    const int exponent = static_cast<int>((magnitude >> 10) & 0x1fu);
    return exponent >= 12 && exponent <= 18;
}

bool validate_model_domain(const GemmCase& test_case, const std::vector<__half>& a,
                           const std::vector<__half>& b) {
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.k; ++col) {
            if (!is_model_value(a[static_cast<std::size_t>(row) * test_case.lda + col])) {
                std::fprintf(stderr, "%s invalid model A input row=%d col=%d\n", test_case.name, row, col);
                return false;
            }
        }
    }
    for (int row = 0; row < test_case.k; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            if (!is_model_value(b[static_cast<std::size_t>(row) * test_case.ldb + col])) {
                std::fprintf(stderr, "%s invalid model B input row=%d col=%d\n", test_case.name, row, col);
                return false;
            }
        }
    }
    return true;
}

bool validate_reduction_domain(const GemmCase& test_case, ComparisonMode comparison) {
    if (valid_reduction_k(test_case.k)) return true;
    std::fprintf(stderr, "%s invalid %s K=%d\n", test_case.name,
                 comparison == ComparisonMode::exact ? "exact" : "model", test_case.k);
    return false;
}

void fill_inputs(const GemmCase& test_case, std::vector<__half>* a, std::vector<__half>* b) {
    std::fill(a->begin(), a->end(), value_from_q(7));
    std::fill(b->begin(), b->end(), value_from_q(7));
    if (is_model_pattern(test_case.pattern)) {
        for (int row = 0; row < test_case.m; ++row) {
            for (int col = 0; col < test_case.k; ++col) {
                const int coordinate = test_case.pattern == Pattern::rounded_cancellation ? col % (test_case.k / 2) : col;
                const bool negate = test_case.pattern == Pattern::rounded_cancellation && col >= test_case.k / 2;
                (*a)[static_cast<std::size_t>(row) * test_case.lda + col] =
                    test_case.pattern == Pattern::rounded_zero ? __float2half(0.0f) :
                    (test_case.pattern == Pattern::rounded_mixed_zeros && col % 4 == 0) ? __float2half(0.0f) :
                    rounded_value(fingerprint(row, coordinate, 17u), test_case.pattern, negate);
            }
        }
        for (int row = 0; row < test_case.k; ++row) {
            for (int col = 0; col < test_case.n; ++col) {
                const int coordinate = test_case.pattern == Pattern::rounded_cancellation ? row % (test_case.k / 2) : row;
                (*b)[static_cast<std::size_t>(row) * test_case.ldb + col] =
                    rounded_value(fingerprint(coordinate, col, 83u), test_case.pattern, false);
            }
        }
        return;
    }
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

int binary16_exponent(__half value) {
    std::uint16_t encoded = 0;
    std::memcpy(&encoded, &value, sizeof(encoded));
    return static_cast<int>((encoded >> 10) & 0x1fu) - 15;
}

bool check_model_fixture_coverage(const GemmCase& test_case, const std::vector<__half>& a,
                                  const std::vector<__half>& b) {
    if (test_case.pattern == Pattern::rounded_mixed && test_case.k == 1024) {
        int min_exponent = 0;
        int max_exponent = 0;
        bool saw_normal = false;
        const auto observe = [&](const __half value) {
            if (__half2float(value) == 0.0f) return;
            const int exponent = binary16_exponent(value);
            min_exponent = saw_normal ? std::min(min_exponent, exponent) : exponent;
            max_exponent = saw_normal ? std::max(max_exponent, exponent) : exponent;
            saw_normal = true;
        };
        for (int row = 0; row < test_case.m; ++row) {
            for (int col = 0; col < test_case.k; ++col) observe(a[static_cast<std::size_t>(row) * test_case.lda + col]);
        }
        for (int row = 0; row < test_case.k; ++row) {
            for (int col = 0; col < test_case.n; ++col) observe(b[static_cast<std::size_t>(row) * test_case.ldb + col]);
        }
        std::printf("%s K=%d exponent-range=[%d,%d]\n", test_case.name, test_case.k, min_exponent, max_exponent);
        if (!saw_normal || min_exponent > -3 || max_exponent < 3) {
            std::fprintf(stderr, "%s lacks mixed exponent coverage\n", test_case.name);
            return false;
        }
    }
    if (test_case.pattern == Pattern::rounded_mixed_zeros) {
        std::size_t zeros = 0;
        std::size_t normals = 0;
        std::size_t b_normals = 0;
        for (int row = 0; row < test_case.m; ++row) {
            for (int k0 = 0; k0 < test_case.k; k0 += 4) {
                for (int offset = 0; offset < 4; ++offset) {
                    const __half value = a[static_cast<std::size_t>(row) * test_case.lda + k0 + offset];
                    const bool zero = __half2float(value) == 0.0f;
                    zeros += zero;
                    normals += !zero;
                    if (zero != (offset == 0)) {
                        std::fprintf(stderr, "%s invalid zero/normal block row=%d k=%d\n", test_case.name, row, k0 + offset);
                        return false;
                    }
                }
            }
        }
        for (int row = 0; row < test_case.k; ++row) {
            for (int col = 0; col < test_case.n; ++col) {
                if (__half2float(b[static_cast<std::size_t>(row) * test_case.ldb + col]) == 0.0f) {
                    std::fprintf(stderr, "%s has a zero B input row=%d col=%d\n", test_case.name, row, col);
                    return false;
                }
                ++b_normals;
            }
        }
        std::printf("%s A zeros=%zu normals=%zu B normals=%zu four-term-blocks=%d\n", test_case.name,
                    zeros, normals, b_normals, test_case.m * (test_case.k / 4));
        if (zeros == 0 || normals == 0 || b_normals == 0) {
            std::fprintf(stderr, "%s lacks zero/normal coverage\n", test_case.name);
            return false;
        }
    }
    return true;
}

std::vector<ReferenceCell> make_reference(const GemmCase& test_case,
                                          const std::vector<__half>& a, const std::vector<__half>& b,
                                          ComparisonMode comparison) {
    std::vector<ReferenceCell> reference(static_cast<std::size_t>(test_case.m) * test_case.ldc,
                                         {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0});
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            double sum = 0.0;
            double sum_abs = 0.0;
            for (int kk = 0; kk < test_case.k; ++kk) {
                const double product = static_cast<double>(__half2float(a[static_cast<std::size_t>(row) * test_case.lda + kk])) *
                                       static_cast<double>(__half2float(b[static_cast<std::size_t>(kk) * test_case.ldb + col]));
                sum += product;
                sum_abs += std::fabs(product);
            }
            const double bound = comparison == ComparisonMode::model
                ? tk_sm7x::test::numerical::error_bound(test_case.k, sum_abs) : 0.0;
            reference[static_cast<std::size_t>(row) * test_case.ldc + col] = {sum, sum_abs, bound};
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

bool check_model_reference(const GemmCase& test_case, const std::vector<ReferenceCell>& reference) {
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test_case.ldc + col;
            const ReferenceCell& cell = reference[index];
            if (!std::isfinite(cell.value) || !std::isfinite(cell.sum_abs) || cell.sum_abs < 0.0 ||
                !std::isfinite(cell.bound) || cell.bound < 0.0 ||
                (cell.sum_abs == 0.0 && cell.bound != 0.0)) {
                std::fprintf(stderr, "%s invalid model reference row=%d col=%d R=%.17g S=%.17g E=%.17g\n",
                             test_case.name, row, col, cell.value, cell.sum_abs, cell.bound);
                return false;
            }
        }
    }
    return true;
}

bool check_double_reference_requirement(const GemmCase& test_case,
                                        const std::vector<ReferenceCell>& reference) {
    if (test_case.pattern != Pattern::rounded_positive) return true;
    std::size_t not_float_representable = 0;
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const double value = reference[static_cast<std::size_t>(row) * test_case.ldc + col].value;
            not_float_representable += static_cast<double>(static_cast<float>(value)) != value;
        }
    }
    const std::size_t total = static_cast<std::size_t>(test_case.m) * test_case.n;
    std::printf("%s non-float-reference=%zu/%zu\n", test_case.name, not_float_representable, total);
    if (not_float_representable == 0) {
        std::fprintf(stderr, "%s lacks a double-only reference cell\n", test_case.name);
        return false;
    }
    return true;
}

double error_ratio(double error, double bound) {
    if (bound == 0.0) return error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return error / bound;
}

struct ComparisonStats {
    std::size_t exact = 0;
    std::size_t different = 0;
    double max_abs = 0.0;
    double max_ratio = 0.0;
    bool within = true;
    int failure_row = -1;
    int failure_col = -1;
    double failure_error = 0.0;
    double failure_bound = 0.0;

    void observe(bool exact_match, double error, double bound, bool accepted, int row, int col) {
        if (exact_match) {
            ++exact;
        } else {
            ++different;
        }
        max_abs = std::max(max_abs, error);
        max_ratio = std::max(max_ratio, error_ratio(error, bound));
        if (!accepted && within) {
            within = false;
            failure_row = row;
            failure_col = col;
            failure_error = error;
            failure_bound = bound;
        }
    }
};

bool check_pair(const GemmCase& test_case, const char* left_name, const std::vector<float>& left,
                const char* right_name, const std::vector<float>& right,
                const std::vector<ReferenceCell>& reference, ComparisonMode comparison) {
    if (comparison == ComparisonMode::exact) {
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
    ComparisonStats stats;
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test_case.ldc + col;
            const double error = std::fabs(static_cast<double>(left[index]) - static_cast<double>(right[index]));
            const double bound = 2.0 * reference[index].bound;
            stats.observe(left[index] == right[index], error, bound,
                          tk_sm7x::test::numerical::within_bound(left[index], right[index], bound), row, col);
        }
    }
    const std::size_t total = static_cast<std::size_t>(test_case.m) * test_case.n;
    std::printf("%s %s/%s exact=%zu/%zu different=%zu/%zu max_abs=%.17g max_ratio=%.17g\n",
                test_case.name, left_name, right_name, stats.exact, total, stats.different, total,
                stats.max_abs, stats.max_ratio);
    if (!stats.within) {
        std::fprintf(stderr, "%s direct model budget mismatch %s/%s row=%d col=%d error=%.17g budget=%.17g\n",
                     test_case.name, left_name, right_name, stats.failure_row, stats.failure_col,
                     stats.failure_error, stats.failure_bound);
    }
    return stats.within;
}

bool check_cpu(const GemmCase& test_case, const char* name,
               const std::vector<float>& actual, const std::vector<ReferenceCell>& reference,
               ComparisonMode comparison) {
    if (comparison == ComparisonMode::exact) {
        for (int row = 0; row < test_case.m; ++row) {
            for (int col = 0; col < test_case.n; ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * test_case.ldc + col;
                if (static_cast<double>(actual[index]) != reference[index].value) {
                    std::fprintf(stderr, "%s %s CPU mismatch row=%d col=%d got=%g want=%g\n",
                                 test_case.name, name, row, col, actual[index], reference[index].value);
                    return false;
                }
            }
        }
        return true;
    }
    ComparisonStats stats;
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test_case.ldc + col;
            const double error = std::fabs(static_cast<double>(actual[index]) - reference[index].value);
            const double bound = reference[index].bound;
            stats.observe(static_cast<double>(actual[index]) == reference[index].value, error, bound,
                          tk_sm7x::test::numerical::within_bound(actual[index], reference[index].value, bound),
                          row, col);
        }
    }
    const std::size_t total = static_cast<std::size_t>(test_case.m) * test_case.n;
    std::printf("%s %s/CPU exact=%zu/%zu different=%zu/%zu max_abs=%.17g max_ratio=%.17g\n",
                test_case.name, name, stats.exact, total, stats.different, total, stats.max_abs, stats.max_ratio);
    if (!stats.within) {
        std::fprintf(stderr, "%s CPU model budget mismatch %s row=%d col=%d error=%.17g budget=%.17g\n",
                     test_case.name, name, stats.failure_row, stats.failure_col, stats.failure_error,
                     stats.failure_bound);
    }
    return stats.within;
}

bool run_case(const GemmCase& test_case, cudaStream_t stream, ComparisonMode comparison) {
    if (!validate_reduction_domain(test_case, comparison)) return false;
    std::vector<__half> a(static_cast<std::size_t>(test_case.m) * test_case.lda);
    std::vector<__half> b(static_cast<std::size_t>(test_case.k) * test_case.ldb);
    std::vector<float> outputs[kOutputCount];
    for (std::vector<float>& output : outputs) output.assign(static_cast<std::size_t>(test_case.m) * test_case.ldc, sentinel());
    fill_inputs(test_case, &a, &b);
    if (comparison == ComparisonMode::exact && !validate_exact_domain(test_case, a, b, true)) return false;
    if (comparison == ComparisonMode::model && !validate_model_domain(test_case, a, b)) return false;
    if (comparison == ComparisonMode::model && !check_model_fixture_coverage(test_case, a, b)) return false;
    const std::vector<ReferenceCell> reference = make_reference(test_case, a, b, comparison);
    if (comparison == ComparisonMode::model && !check_double_reference_requirement(test_case, reference)) return false;
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
    if (comparison == ComparisonMode::model) matched = check_model_reference(test_case, reference) && matched;
    int pair_count = 0;
    for (int left = 0; left < kOutputCount; ++left) for (int right = left + 1; right < kOutputCount; ++right) {
        matched = check_pair(test_case, names[left], outputs[left], names[right], outputs[right], reference, comparison) && matched;
        ++pair_count;
    }
    if (pair_count != kExpectedPairCount) {
        std::fprintf(stderr, "%s expected %d direct backend pairs, got %d\n", test_case.name, kExpectedPairCount, pair_count);
        matched = false;
    }
    for (int index = 0; index < kOutputCount; ++index) {
        matched = check_cpu(test_case, names[index], outputs[index], reference, comparison) && matched;
    }
    if (matched) {
        std::printf("%s: direct pairs=%d, CPU comparisons=%d %s PASS\n", test_case.name, pair_count,
                    kOutputCount, comparison == ComparisonMode::model ? "model-budget" : "exact");
    }
    return release() && matched;
}

}  // namespace

int main() {
    if (!run_exact_domain_self_checks()) return EXIT_FAILURE;
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
    for (const GemmCase& test_case : test_cases) passed = run_case(test_case, stream, ComparisonMode::exact) && passed;
    const GemmCase model_cases[] = {
        {"rounded-positive-short", 16, 16, 16, 24, 24, 20, Pattern::rounded_positive},
        {"rounded-positive-long", 16, 16, 1024, 1032, 24, 20, Pattern::rounded_positive},
        {"rounded-signed", 32, 48, 256, 264, 56, 52, Pattern::rounded_signed},
        {"rounded-mixed", 32, 48, 256, 264, 56, 52, Pattern::rounded_mixed},
        {"rounded-mixed-long", 32, 48, 1024, 1032, 56, 52, Pattern::rounded_mixed},
        {"rounded-mixed-zeros", 32, 48, 256, 264, 56, 52, Pattern::rounded_mixed_zeros},
        {"rounded-cancellation", 16, 32, 1024, 1032, 40, 36, Pattern::rounded_cancellation},
        {"rounded-zero", 16, 16, 32, 40, 24, 20, Pattern::rounded_zero},
        {"rounded-production-small", 256, 384, 64, 72, 392, 388, Pattern::rounded_mixed},
        {"rounded-production-large", 1024, 1024, 32, 40, 1032, 1028, Pattern::rounded_mixed},
    };
    for (const GemmCase& test_case : model_cases) passed = run_case(test_case, stream, ComparisonMode::model) && passed;
    const bool destroyed = tk_sm7x::test::cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return passed && destroyed ? EXIT_SUCCESS : EXIT_FAILURE;
}
