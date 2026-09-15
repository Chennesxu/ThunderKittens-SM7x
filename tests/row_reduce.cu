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

#include "row_reduce_reference.cuh"
#include "test_utils.cuh"
#include "tk_sm7x/mma.cuh"
#include "tk_sm7x/ptx_backend.cuh"

namespace {

using tk_sm7x::detail::row_max_op;
using tk_sm7x::detail::row_sum_op;
using tk_sm7x::test::canonical_row;
using tk_sm7x::test::cuda_ok;
using tk_sm7x::test::reference_row_reduce;

constexpr int kTile = 16;
constexpr int kCells = kTile * kTile;
constexpr std::uint32_t kSentinelBits = 0x7fc00000u;

template <class Backend>
__device__ __forceinline__ void accumulate(
    int k, const __half* a, int lda, const __half* b, int ldb,
    typename Backend::accumulator& accumulator, __half* as, __half* bs) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    Backend::clear(accumulator);
    for (std::size_t k0 = 0; k0 < static_cast<std::size_t>(k); k0 += kTile) {
        for (int linear = lane; linear < kCells; linear += 32) {
            as[linear] = a[static_cast<std::size_t>(linear / kTile) *
                               static_cast<std::size_t>(lda) +
                           k0 + static_cast<std::size_t>(linear % kTile)];
            bs[static_cast<std::size_t>(linear % kTile) * kTile + linear / kTile] =
                b[(k0 + static_cast<std::size_t>(linear / kTile)) *
                      static_cast<std::size_t>(ldb) +
                  static_cast<std::size_t>(linear % kTile)];
        }
        __syncwarp(0xffffffffu);
        typename Backend::fragment_a af;
        typename Backend::fragment_b bf;
        Backend::load_a(af, as, kTile);
        Backend::load_b(bf, bs, kTile);
        Backend::mma(accumulator, af, bf);
        __syncwarp(0xffffffffu);
    }
}

template <class Backend>
__device__ __forceinline__ void stage_and_reference(
    const typename Backend::accumulator& accumulator, float* tile,
    float* shared_sum, float* shared_max, float* cs) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    Backend::store(cs, accumulator, kTile);
    __syncwarp(0xffffffffu);
    reference_row_reduce<row_sum_op>(shared_sum, cs);
    reference_row_reduce<row_max_op>(shared_max, cs);
    for (int linear = lane; linear < kCells; linear += 32) tile[linear] = cs[linear];
}

extern "C" __global__ void row_reduce_wmma(
    int k, const __half* a, int lda, const __half* b, int ldb, float* tile,
    float* shared_sum, float* shared_max) {
    __shared__ __align__(32) __half as[kCells];
    __shared__ __align__(32) __half bs[kCells];
    __shared__ __align__(32) float cs[kCells];
    using backend = tk_sm7x::detail::warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::target>;
    backend::accumulator accumulator;
    accumulate<backend>(k, a, lda, b, ldb, accumulator, as, bs);
    stage_and_reference<backend>(accumulator, tile, shared_sum, shared_max, cs);
}

extern "C" __global__ void row_reduce_m8(
    int k, const __half* a, int lda, const __half* b, int ldb, float* tile,
    float* register_sum, float* register_max, float* shared_sum, float* shared_max) {
    __shared__ __align__(32) __half as[kCells];
    __shared__ __align__(32) __half bs[kCells];
    __shared__ __align__(32) float cs[kCells];
    using backend = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm70>;
    backend::accumulator accumulator;
    accumulate<backend>(k, a, lda, b, ldb, accumulator, as, bs);
    backend::row_sum(register_sum, accumulator);
    backend::row_max(register_max, accumulator);
    stage_and_reference<backend>(accumulator, tile, shared_sum, shared_max, cs);
}

#if defined(KITTENS_SM75)
extern "C" __global__ void row_reduce_m16(
    int k, const __half* a, int lda, const __half* b, int ldb, float* tile,
    float* register_sum, float* register_max, float* shared_sum, float* shared_max) {
    __shared__ __align__(32) __half as[kCells];
    __shared__ __align__(32) __half bs[kCells];
    __shared__ __align__(32) float cs[kCells];
    using backend = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm75>;
    backend::accumulator accumulator;
    accumulate<backend>(k, a, lda, b, ldb, accumulator, as, bs);
    backend::row_sum(register_sum, accumulator);
    backend::row_max(register_max, accumulator);
    stage_and_reference<backend>(accumulator, tile, shared_sum, shared_max, cs);
}
#endif

enum class Domain { exact, rounded };

struct Fixture {
    const char* name;
    int k;
    int lda;
    int ldb;
    Domain domain;
};

enum class Kind {
    wmma,
    m8,
#if defined(KITTENS_SM75)
    m16,
#endif
};

struct Backend {
    const char* name;
    Kind kind;
};

bool has_register_path(Kind kind) { return kind != Kind::wmma; }

float sentinel() {
    float value = 0.0f;
    std::memcpy(&value, &kSentinelBits, sizeof(value));
    return value;
}

__half exact_value(int row, int col, int seed) {
    return __float2half(static_cast<float>((row * 3 + col * 5 + seed) % 17 - 8) / 8.0f);
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

__half rounded_value(std::uint32_t hash) {
    const int exponent = static_cast<int>((hash >> 10) % 7) - 3;
    float value = std::ldexp(1.0f + static_cast<float>(hash & 1023u) / 1024.0f, exponent);
    if ((hash >> 31) != 0u) value = -value;
    return __float2half(value);
}

bool is_exact_value(__half value) {
    const float logical = __half2float(value);
    return std::isfinite(logical) && std::fabs(logical) <= 1.0f &&
           std::trunc(8.0f * logical) == 8.0f * logical;
}

bool is_rounded_value(__half value) {
    std::uint16_t encoded = 0;
    std::memcpy(&encoded, &value, sizeof(encoded));
    const int exponent = static_cast<int>((encoded >> 10) & 0x1fu);
    return exponent >= 12 && exponent <= 18;
}

void fill_inputs(const Fixture& fixture, std::vector<__half>* a, std::vector<__half>* b) {
    for (int row = 0; row < kTile; ++row) {
        for (int col = 0; col < fixture.k; ++col) {
            (*a)[static_cast<std::size_t>(row) * fixture.lda + col] =
                fixture.domain == Domain::exact
                    ? exact_value(row, col, 1)
                    : rounded_value(fingerprint(row, col, 17u));
        }
    }
    for (int row = 0; row < fixture.k; ++row) {
        for (int col = 0; col < kTile; ++col) {
            (*b)[static_cast<std::size_t>(row) * fixture.ldb + col] =
                fixture.domain == Domain::exact
                    ? exact_value(row, col, 9)
                    : rounded_value(fingerprint(row, col, 83u));
        }
    }
}

bool validate_domain(const Fixture& fixture, const std::vector<__half>& a,
                     const std::vector<__half>& b) {
    if (fixture.k <= 0 || fixture.k % kTile != 0 || fixture.k > 1024) {
        std::fprintf(stderr, "%s invalid K=%d\n", fixture.name, fixture.k);
        return false;
    }
    for (int row = 0; row < kTile; ++row) {
        for (int col = 0; col < fixture.k; ++col) {
            const __half value = a[static_cast<std::size_t>(row) * fixture.lda + col];
            const bool ok = fixture.domain == Domain::exact ? is_exact_value(value)
                                                            : is_rounded_value(value);
            if (!ok) {
                std::fprintf(stderr, "%s invalid A input row=%d col=%d\n", fixture.name, row, col);
                return false;
            }
        }
    }
    for (int row = 0; row < fixture.k; ++row) {
        for (int col = 0; col < kTile; ++col) {
            const __half value = b[static_cast<std::size_t>(row) * fixture.ldb + col];
            const bool ok = fixture.domain == Domain::exact ? is_exact_value(value)
                                                            : is_rounded_value(value);
            if (!ok) {
                std::fprintf(stderr, "%s invalid B input row=%d col=%d\n", fixture.name, row, col);
                return false;
            }
        }
    }
    return true;
}

// Products are multiples of 2^-26 with magnitude below 256 and a row consumes at
// most 16*1024 of them, so the scaled magnitude stays below 2^48 and every
// reference value below is exact in binary64 for both declared domains.
void exact_reference(const Fixture& fixture, const std::vector<__half>& a,
                     const std::vector<__half>& b, std::vector<double>* row_sum,
                     std::vector<double>* row_max) {
    row_sum->assign(kTile, 0.0);
    row_max->assign(kTile, 0.0);
    for (int row = 0; row < kTile; ++row) {
        double sum = 0.0;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < kTile; ++col) {
            double cell = 0.0;
            for (int kk = 0; kk < fixture.k; ++kk) {
                cell += static_cast<double>(
                            __half2float(a[static_cast<std::size_t>(row) * fixture.lda + kk])) *
                        static_cast<double>(
                            __half2float(b[static_cast<std::size_t>(kk) * fixture.ldb + col]));
            }
            sum += cell;
            maximum = std::max(maximum, cell);
        }
        (*row_sum)[row] = sum;
        (*row_max)[row] = maximum;
    }
}

bool check_written(const char* label, const std::vector<float>& values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            std::fprintf(stderr, "%s unwritten/nonfinite entry %zu\n", label, index);
            return false;
        }
    }
    return true;
}

bool check_equal(const char* label, const char* left_name, const std::vector<float>& left,
                 const char* right_name, const std::vector<float>& right) {
    for (int row = 0; row < kTile; ++row) {
        if (left[row] != right[row]) {
            std::fprintf(stderr, "%s %s/%s row=%d %s=%.17g %s=%.17g\n", label, left_name,
                         right_name, row, left_name, static_cast<double>(left[row]), right_name,
                         static_cast<double>(right[row]));
            return false;
        }
    }
    return true;
}

bool check_exact(const char* label, const char* name, const std::vector<float>& actual,
                 const std::vector<double>& reference) {
    for (int row = 0; row < kTile; ++row) {
        if (static_cast<double>(actual[row]) != reference[row]) {
            std::fprintf(stderr, "%s %s exact mismatch row=%d got=%.17g want=%.17g\n", label, name,
                         row, static_cast<double>(actual[row]), reference[row]);
            return false;
        }
    }
    return true;
}

std::size_t rounded_rows(const std::vector<float>& actual, const std::vector<double>& reference) {
    std::size_t count = 0;
    for (int row = 0; row < kTile; ++row) {
        count += static_cast<double>(actual[row]) != reference[row];
    }
    return count;
}

struct Buffers {
    float* tile = nullptr;
    float* register_sum = nullptr;
    float* register_max = nullptr;
    float* shared_sum = nullptr;
    float* shared_max = nullptr;
};

bool run_backend(const Fixture& fixture, const Backend& backend, const __half* device_a,
                 const __half* device_b, const std::vector<double>& exact_sum,
                 const std::vector<double>& exact_max, cudaStream_t stream) {
    char label[128];
    std::snprintf(label, sizeof(label), "%s %s", fixture.name, backend.name);
    const bool register_path = has_register_path(backend.kind);
    Buffers device;
    const std::size_t tile_bytes = kCells * sizeof(float);
    const std::size_t row_bytes = kTile * sizeof(float);
    const auto release = [&]() {
        bool ok = true;
        for (float* pointer : {device.tile, device.register_sum, device.register_max,
                               device.shared_sum, device.shared_max}) {
            if (pointer != nullptr) ok = cuda_ok(cudaFree(pointer), "cudaFree") && ok;
        }
        return ok;
    };
    bool prepared = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device.tile), tile_bytes),
                            "cudaMalloc(tile)");
    for (float** pointer : {&device.shared_sum, &device.shared_max}) {
        prepared = cuda_ok(cudaMalloc(reinterpret_cast<void**>(pointer), row_bytes),
                           "cudaMalloc(rows)") && prepared;
    }
    if (register_path) {
        for (float** pointer : {&device.register_sum, &device.register_max}) {
            prepared = cuda_ok(cudaMalloc(reinterpret_cast<void**>(pointer), row_bytes),
                               "cudaMalloc(rows)") && prepared;
        }
    }
    if (!prepared) return release() && false;

    const std::vector<float> tile_sentinel(kCells, sentinel());
    const std::vector<float> row_sentinel(kTile, sentinel());
    prepared = cuda_ok(cudaMemcpyAsync(device.tile, tile_sentinel.data(), tile_bytes,
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(tile sentinel)");
    for (float* pointer : {device.register_sum, device.register_max, device.shared_sum,
                           device.shared_max}) {
        if (pointer == nullptr) continue;
        prepared = cuda_ok(cudaMemcpyAsync(pointer, row_sentinel.data(), row_bytes,
                                           cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync(row sentinel)") && prepared;
    }
    if (!prepared) return release() && false;

    switch (backend.kind) {
        case Kind::wmma:
            row_reduce_wmma<<<1, 32, 0, stream>>>(fixture.k, device_a, fixture.lda, device_b,
                                                  fixture.ldb, device.tile, device.shared_sum,
                                                  device.shared_max);
            break;
        case Kind::m8:
            row_reduce_m8<<<1, 32, 0, stream>>>(fixture.k, device_a, fixture.lda, device_b,
                                                fixture.ldb, device.tile, device.register_sum,
                                                device.register_max, device.shared_sum,
                                                device.shared_max);
            break;
#if defined(KITTENS_SM75)
        case Kind::m16:
            row_reduce_m16<<<1, 32, 0, stream>>>(fixture.k, device_a, fixture.lda, device_b,
                                                 fixture.ldb, device.tile, device.register_sum,
                                                 device.register_max, device.shared_sum,
                                                 device.shared_max);
            break;
#endif
    }
    if (!cuda_ok(cudaGetLastError(), "row reduce launch")) return release() && false;

    std::vector<float> tile(kCells);
    std::vector<float> register_sum(kTile);
    std::vector<float> register_max(kTile);
    std::vector<float> shared_sum(kTile);
    std::vector<float> shared_max(kTile);
    bool copied = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    copied = cuda_ok(cudaMemcpy(tile.data(), device.tile, tile_bytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy(tile)") && copied;
    copied = cuda_ok(cudaMemcpy(shared_sum.data(), device.shared_sum, row_bytes,
                                cudaMemcpyDeviceToHost), "cudaMemcpy(shared_sum)") && copied;
    copied = cuda_ok(cudaMemcpy(shared_max.data(), device.shared_max, row_bytes,
                                cudaMemcpyDeviceToHost), "cudaMemcpy(shared_max)") && copied;
    if (register_path) {
        copied = cuda_ok(cudaMemcpy(register_sum.data(), device.register_sum, row_bytes,
                                    cudaMemcpyDeviceToHost), "cudaMemcpy(register_sum)") && copied;
        copied = cuda_ok(cudaMemcpy(register_max.data(), device.register_max, row_bytes,
                                    cudaMemcpyDeviceToHost), "cudaMemcpy(register_max)") && copied;
    }
    if (!copied) return release() && false;

    bool matched = check_written(label, tile) && check_written(label, shared_sum) &&
                   check_written(label, shared_max);
    if (register_path) {
        matched = check_written(label, register_sum) && matched;
        matched = check_written(label, register_max) && matched;
    }

    std::vector<float> host_sum(kTile);
    std::vector<float> host_max(kTile);
    for (int row = 0; row < kTile; ++row) {
        host_sum[row] = canonical_row<row_sum_op>(tile.data() + static_cast<std::size_t>(row) * kTile);
        host_max[row] = canonical_row<row_max_op>(tile.data() + static_cast<std::size_t>(row) * kTile);
    }

    matched = check_equal(label, "shared-sum", shared_sum, "host-sum", host_sum) && matched;
    matched = check_equal(label, "shared-max", shared_max, "host-max", host_max) && matched;
    if (register_path) {
        matched = check_equal(label, "register-sum", register_sum, "shared-sum", shared_sum) && matched;
        matched = check_equal(label, "register-max", register_max, "shared-max", shared_max) && matched;
        matched = check_equal(label, "register-sum", register_sum, "host-sum", host_sum) && matched;
        matched = check_equal(label, "register-max", register_max, "host-max", host_max) && matched;
    }

    if (fixture.domain == Domain::exact) {
        matched = check_exact(label, "shared-sum", shared_sum, exact_sum) && matched;
        matched = check_exact(label, "shared-max", shared_max, exact_max) && matched;
        if (register_path) {
            matched = check_exact(label, "register-sum", register_sum, exact_sum) && matched;
            matched = check_exact(label, "register-max", register_max, exact_max) && matched;
        }
    } else {
        const std::vector<float>& observed = register_path ? register_sum : shared_sum;
        const std::size_t rounded = rounded_rows(observed, exact_sum);
        std::printf("%s rounded-rows=%zu/%d\n", label, rounded, kTile);
        if (rounded == 0) {
            std::fprintf(stderr, "%s exercises no FP32 rounding\n", label);
            matched = false;
        }
    }

    if (matched) {
        std::printf("%s: %s PASS\n", label,
                    register_path ? "register/shared/host" : "shared/host");
    }
    return release() && matched;
}

bool run_fixture(const Fixture& fixture, cudaStream_t stream) {
    std::vector<__half> a(static_cast<std::size_t>(kTile) * fixture.lda, __float2half(0.0f));
    std::vector<__half> b(static_cast<std::size_t>(fixture.k) * fixture.ldb, __float2half(0.0f));
    fill_inputs(fixture, &a, &b);
    if (!validate_domain(fixture, a, b)) return false;
    std::vector<double> exact_sum;
    std::vector<double> exact_max;
    exact_reference(fixture, a, b, &exact_sum, &exact_max);

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    const std::size_t a_bytes = a.size() * sizeof(__half);
    const std::size_t b_bytes = b.size() * sizeof(__half);
    const auto release = [&]() {
        bool ok = true;
        if (device_a != nullptr) ok = cuda_ok(cudaFree(device_a), "cudaFree(A)") && ok;
        if (device_b != nullptr) ok = cuda_ok(cudaFree(device_b), "cudaFree(B)") && ok;
        return ok;
    };
    bool prepared = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_a), a_bytes),
                            "cudaMalloc(A)") &&
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_b), b_bytes),
                            "cudaMalloc(B)");
    prepared = cuda_ok(cudaMemcpyAsync(device_a, a.data(), a_bytes, cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(A)") && prepared;
    prepared = cuda_ok(cudaMemcpyAsync(device_b, b.data(), b_bytes, cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(B)") && prepared;
    if (!prepared) return release() && false;

    const Backend backends[] = {
        {"wmma", Kind::wmma},
        {"m8", Kind::m8},
#if defined(KITTENS_SM75)
        {"m16", Kind::m16},
#endif
    };
    bool matched = true;
    for (const Backend& backend : backends) {
        matched = run_backend(fixture, backend, device_a, device_b, exact_sum, exact_max, stream) &&
                  matched;
    }
    return release() && matched;
}

constexpr Fixture kFixtures[] = {
    {"exact-k16", 16, 24, 24, Domain::exact},
    {"exact-k64", 64, 72, 24, Domain::exact},
    {"rounded-k16", 16, 24, 24, Domain::rounded},
    {"rounded-k256", 256, 264, 24, Domain::rounded},
    {"rounded-k1024", 1024, 1032, 24, Domain::rounded},
};

constexpr bool roster_has_rounded_case() {
    for (const Fixture& fixture : kFixtures) {
        if (fixture.domain == Domain::rounded) return true;
    }
    return false;
}

constexpr bool roster_has_exact_case() {
    for (const Fixture& fixture : kFixtures) {
        if (fixture.domain == Domain::exact) return true;
    }
    return false;
}

static_assert(roster_has_exact_case(),
              "the roster needs an exact-domain case for the order-free reference");
static_assert(roster_has_rounded_case(),
              "the roster needs a rounding-prone case for the declared order");

}  // namespace

int main() {
    int device = -1;
    const int selection = tk_sm7x::test::select_sm75_device(&device);
    if (selection != EXIT_SUCCESS) return selection;
    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) return EXIT_FAILURE;
    bool passed = true;
    for (const Fixture& fixture : kFixtures) passed = run_fixture(fixture, stream) && passed;
    const bool destroyed = cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy");
    if (passed && destroyed) std::printf("row reduce: PASS ordinal=%d\n", device);
    return passed && destroyed ? EXIT_SUCCESS : EXIT_FAILURE;
}
