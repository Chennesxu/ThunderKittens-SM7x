#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "row_reduce_reference.cuh"
#include "test_utils.cuh"
#include "tk_sm7x/reduce.cuh"

namespace {

constexpr int kTile = 16;
constexpr int kCells = 256;
constexpr int kGuardedRows = 20;

struct Sum {
    __host__ __device__ float operator()(float a, float b) const { return a + b; }
};

struct Max {
    __host__ __device__ float operator()(float a, float b) const { return a > b ? a : b; }
};

__global__ void tile_reduce_kernel(
    const __half* a, const __half* b, int lda, int ldb, int warps,
    bool shared_destination, float* before, float* after, float* sums,
    float* maxima, float* cross_sum, float* cross_max, float* sums_again,
    float* maxima_again) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const std::size_t group = static_cast<std::size_t>(blockIdx.x) * warps + warp;
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::row_major> as[4];
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::col_major> bs[4];
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> scratch[4];
    __shared__ float shared_sum[4][16];
    __shared__ float shared_max[4][16];
    const std::size_t a_base = group * static_cast<std::size_t>(kTile) * lda;
    const std::size_t b_base = group * static_cast<std::size_t>(kTile) * ldb;
    for (int linear = lane; linear < kCells; linear += 32) {
        const int row = linear / kTile;
        const int col = linear % kTile;
        as[warp].data[linear] = a[a_base + static_cast<std::size_t>(row) * lda + col];
        bs[warp].data[static_cast<std::size_t>(col) * kTile + row] =
            b[b_base + static_cast<std::size_t>(row) * ldb + col];
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
    for (int linear = lane; linear < kCells; linear += 32) {
        before[group * kCells + linear] = scratch[warp].data[linear];
    }
    __syncwarp(0xffffffffu);

    float* sum_dst = shared_destination ? shared_sum[warp] : sums + group * kGuardedRows + 2;
    float* max_dst = shared_destination ? shared_max[warp] : maxima + group * kGuardedRows + 2;
    tk_sm7x::row_sum(sum_dst, cf, scratch[warp]);
    if (lane < kTile) {
        const int other = (lane + 7) % kTile;
        cross_sum[group * kGuardedRows + 2 + lane] = sum_dst[other];
        if (shared_destination) sums[group * kGuardedRows + 2 + lane] = sum_dst[lane];
    }
    tk_sm7x::row_max(max_dst, cf, scratch[warp]);
    if (lane < kTile) {
        const int other = (lane + 7) % kTile;
        cross_max[group * kGuardedRows + 2 + lane] = max_dst[other];
        if (shared_destination) maxima[group * kGuardedRows + 2 + lane] = max_dst[lane];
    }
    tk_sm7x::row_sum(sums_again + group * kGuardedRows + 2, cf, scratch[warp]);
    tk_sm7x::row_max(maxima_again + group * kGuardedRows + 2, cf, scratch[warp]);
    tk_sm7x::store(scratch[warp], cf);
    __syncwarp(0xffffffffu);
    for (int linear = lane; linear < kCells; linear += 32) {
        after[group * kCells + linear] = scratch[warp].data[linear];
    }
}

float sentinel() {
    const unsigned bits = 0x7fc00000u;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::size_t index(int group, int row, int ld) {
    return static_cast<std::size_t>(group) * kTile * ld +
           static_cast<std::size_t>(row) * ld;
}

__half rounded_value(unsigned seed) {
    seed ^= seed >> 16;
    seed *= 0x7feb352du;
    seed ^= seed >> 15;
    seed *= 0x846ca68bu;
    seed ^= seed >> 16;
    float value = std::ldexp(1.0f + static_cast<float>(seed & 1023u) / 1024.0f,
                             static_cast<int>((seed >> 10) % 7) - 3);
    if ((seed >> 31) != 0u) value = -value;
    return __float2half(value);
}

void fill(int groups, int lda, int ldb, int kind,
          std::vector<__half>* a, std::vector<__half>* b) {
    for (int group = 0; group < groups; ++group) {
        for (int row = 0; row < kTile; ++row) {
            for (int col = 0; col < kTile; ++col) {
                int av = (row * 3 + col * 5 + group) % 13 - 6;
                int bv = (row * 7 + col * 2 + group) % 11 - 5;
                if (kind == 2) {
                    av = -(1 + (row + col + group) % 5);
                    bv = 1 + (row * 2 + col + group) % 3;
                }
                if (kind == 3) av = 0;
                if (kind == 1) {
                    const unsigned seed = static_cast<unsigned>(row) * 0x9e3779b9u ^
                        static_cast<unsigned>(col) * 0x85ebca6bu ^
                        static_cast<unsigned>(group) * 0xc2b2ae35u;
                    (*a)[index(group, row, lda) + col] = rounded_value(seed ^ 17u);
                    (*b)[index(group, row, ldb) + col] = rounded_value(seed ^ 83u);
                } else {
                    (*a)[index(group, row, lda) + col] = __float2half(av / 8.0f);
                    (*b)[index(group, row, ldb) + col] = __float2half(bv / 8.0f);
                }
            }
        }
    }
}

bool run_case(const char* name, int warps, int blocks, int lda, int ldb,
              int kind, bool shared_destination) {
    const int groups = warps * blocks;
    const std::size_t cells = static_cast<std::size_t>(groups) * kCells;
    const std::size_t rows = static_cast<std::size_t>(groups) * kGuardedRows;
    std::vector<__half> host_a(static_cast<std::size_t>(groups) * kTile * lda);
    std::vector<__half> host_b(static_cast<std::size_t>(groups) * kTile * ldb);
    fill(groups, lda, ldb, kind, &host_a, &host_b);
    __half *device_a = nullptr, *device_b = nullptr;
    float *before = nullptr, *after = nullptr, *sums = nullptr, *maxima = nullptr;
    float *cross_sum = nullptr, *cross_max = nullptr, *sums_again = nullptr;
    float* maxima_again = nullptr;
    bool ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(&device_a, host_a.size() * sizeof(__half)), "cudaMallocManaged(A)");
    ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(&device_b, host_b.size() * sizeof(__half)), "cudaMallocManaged(B)") && ok;
    for (float** p : {&before, &after}) {
        ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(p, cells * sizeof(float)), "cudaMallocManaged(tile)") && ok;
    }
    for (float** p : {&sums, &maxima, &cross_sum, &cross_max, &sums_again, &maxima_again}) {
        ok = tk_sm7x::test::cuda_ok(cudaMallocManaged(p, rows * sizeof(float)), "cudaMallocManaged(rows)") && ok;
    }
    if (ok) {
        std::memcpy(device_a, host_a.data(), host_a.size() * sizeof(__half));
        std::memcpy(device_b, host_b.data(), host_b.size() * sizeof(__half));
        for (float* p : {before, after}) for (std::size_t i = 0; i < cells; ++i) p[i] = sentinel();
        for (float* p : {sums, maxima, cross_sum, cross_max, sums_again, maxima_again}) {
            for (std::size_t i = 0; i < rows; ++i) p[i] = sentinel();
        }
        tile_reduce_kernel<<<blocks, warps * 32>>>(device_a, device_b, lda, ldb, warps,
            shared_destination, before, after, sums, maxima, cross_sum, cross_max,
            sums_again, maxima_again);
        ok = tk_sm7x::test::cuda_ok(cudaGetLastError(), "tile_reduce_kernel launch") && ok;
        ok = tk_sm7x::test::cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") && ok;
    }
    int roundings = 0;
    if (ok) {
        for (int group = 0; group < groups; ++group) {
            for (int row = 0; row < kTile; ++row) {
                const float* tile_row = before + static_cast<std::size_t>(group) * kCells + row * kTile;
                Sum sum_op;
                Max max_op;
                const float expected_sum = tk_sm7x::test::canonical_row_tree(sum_op, tile_row);
                const float expected_max = tk_sm7x::test::canonical_row_tree(max_op, tile_row);
                const std::size_t pos = static_cast<std::size_t>(group) * kGuardedRows + 2 + row;
                const std::size_t other = static_cast<std::size_t>(group) * kGuardedRows + 2 + (row + 7) % kTile;
                if (sums[pos] != expected_sum || maxima[pos] != expected_max ||
                    sums_again[pos] != expected_sum || maxima_again[pos] != expected_max ||
                    cross_sum[pos] != sums[other] || cross_max[pos] != maxima[other]) {
                    std::fprintf(stderr, "%s group=%d row=%d sum=%.17g expected=%.17g max=%.17g expected=%.17g\n",
                        name, group, row, static_cast<double>(sums[pos]),
                        static_cast<double>(expected_sum), static_cast<double>(maxima[pos]),
                        static_cast<double>(expected_max));
                    ok = false;
                }
                if (kind == 0 || kind == 2 || kind == 3) {
                    double exact_sum = 0.0;
                    double exact_max = -1.0e300;
                    for (int col = 0; col < kTile; ++col) {
                        double cell = 0.0;
                        for (int kk = 0; kk < kTile; ++kk) {
                            cell += static_cast<double>(__half2float(host_a[index(group, row, lda) + kk])) *
                                    static_cast<double>(__half2float(host_b[index(group, kk, ldb) + col]));
                        }
                        exact_sum += cell;
                        if (cell > exact_max) exact_max = cell;
                    }
                    if (static_cast<double>(sums[pos]) != exact_sum ||
                        static_cast<double>(maxima[pos]) != exact_max) ok = false;
                }
                if (kind == 2 && !(maxima[pos] < 0.0f)) ok = false;
                if (kind == 3 && (sums[pos] != 0.0f || maxima[pos] != 0.0f)) ok = false;
                tk_sm7x::test::rounding_census_combine census;
                tk_sm7x::test::canonical_row_tree(census, tile_row);
                roundings += census.inexact;
            }
            for (int linear = 0; linear < kCells; ++linear) {
                const std::size_t pos = static_cast<std::size_t>(group) * kCells + linear;
                if (!std::isfinite(before[pos]) || before[pos] != after[pos]) ok = false;
            }
            for (float* p : {sums, maxima, cross_sum, cross_max, sums_again, maxima_again}) {
                const float guard_value = sentinel();
                for (int guard : {0, 1, 18, 19}) {
                    if (std::memcmp(&p[static_cast<std::size_t>(group) * kGuardedRows + guard],
                                    &guard_value, sizeof(float)) != 0) ok = false;
                }
            }
        }
        if (kind == 1 && roundings == 0) ok = false;
        std::printf("%s: %s tree-roundings=%d\n", name, ok ? "PASS" : "FAIL", roundings);
    }
    for (void* p : {static_cast<void*>(device_a), static_cast<void*>(device_b),
                    static_cast<void*>(before), static_cast<void*>(after),
                    static_cast<void*>(sums), static_cast<void*>(maxima),
                    static_cast<void*>(cross_sum), static_cast<void*>(cross_max),
                    static_cast<void*>(sums_again), static_cast<void*>(maxima_again)}) {
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
    ok = run_case("exact-one-shared", 1, 1, 16, 16, 0, true) && ok;
    ok = run_case("rounded-four-global", 4, 2, 19, 21, 1, false) && ok;
    ok = run_case("negative-four-shared", 4, 2, 19, 21, 2, true) && ok;
    ok = run_case("zero-one-global", 1, 2, 16, 16, 3, false) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
