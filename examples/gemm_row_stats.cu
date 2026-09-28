#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "tk_sm7x/reduce.cuh"

namespace {

constexpr int kTile = 16;
constexpr int kCells = kTile * kTile;

__global__ void gemm_row_stats(const __half* a, const __half* b,
                               float* c, float* sums, float* maxima) {
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::row_major> shared_a;
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::col_major> shared_b;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> shared_c;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> scratch;

    tk_sm7x::load_block<32>(shared_a, tk_sm7x::gl<const __half>{a, 16, 16, 16}, 0, 0);
    tk_sm7x::load_block<32>(shared_b, tk_sm7x::gl<const __half>{b, 16, 16, 16}, 0, 0);
    __syncthreads();

    tk_sm7x::rt_a a_tile;
    tk_sm7x::rt_b b_tile;
    tk_sm7x::rt_c c_tile;
    tk_sm7x::load(a_tile, shared_a, 0);
    tk_sm7x::load(b_tile, shared_b, 0);
    tk_sm7x::zero(c_tile);
    tk_sm7x::mma(c_tile, a_tile, b_tile);
    tk_sm7x::store(shared_c, c_tile);
    __syncwarp();
    tk_sm7x::store_block<32>(tk_sm7x::gl<float>{c, 16, 16, 16}, shared_c, 0, 0);
    tk_sm7x::row_sum(sums, c_tile, scratch);
    tk_sm7x::row_max(maxima, c_tile, scratch);
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

int select_sm75_device() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        const bool unavailable = status == cudaErrorNoDevice ||
                                 status == cudaErrorInsufficientDriver ||
                                 status == cudaErrorSystemDriverMismatch;
        std::fprintf(stderr, "%s: cudaGetDeviceCount failed: %s\n",
                     unavailable ? "SKIP" : "ERROR", cudaGetErrorString(status));
        return unavailable ? 77 : EXIT_FAILURE;
    }
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cudaDeviceProp properties{};
        if (!cuda_ok(cudaGetDeviceProperties(&properties, ordinal),
                     "cudaGetDeviceProperties")) return EXIT_FAILURE;
        if (properties.major == 7 && properties.minor == 5) {
            if (!cuda_ok(cudaSetDevice(ordinal), "cudaSetDevice")) return EXIT_FAILURE;
            std::printf("Selected CUDA device ordinal=%d name=%s cc=%d.%d\n",
                        ordinal, properties.name, properties.major, properties.minor);
            return EXIT_SUCCESS;
        }
    }
    std::fprintf(stderr, "SKIP: no CUDA device with compute capability 7.5\n");
    return 77;
}

int numerator_a(int row, int col) {
    return (row * 3 + col * 5) % 13 - 6;
}

int numerator_b(int row, int col) {
    return (row * 7 + col * 2) % 11 - 5;
}

bool check_outputs(const float* c, const float* sums, const float* maxima) {
    bool ok = true;
    for (int row = 0; row < kTile; ++row) {
        double expected_sum = 0.0;
        double expected_max = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < kTile; ++col) {
            double expected = 0.0;
            for (int k = 0; k < kTile; ++k) {
                expected += static_cast<double>(numerator_a(row, k)) *
                            static_cast<double>(numerator_b(k, col)) / 64.0;
            }
            const std::size_t index = static_cast<std::size_t>(row) * kTile + col;
            if (!std::isfinite(c[index]) || static_cast<double>(c[index]) != expected) {
                std::fprintf(stderr, "C[%d,%d] got=%.17g expected=%.17g\n",
                             row, col, static_cast<double>(c[index]), expected);
                ok = false;
            }
            expected_sum += expected;
            if (expected > expected_max) expected_max = expected;
        }
        if (!std::isfinite(sums[row]) ||
            static_cast<double>(sums[row]) != expected_sum) {
            std::fprintf(stderr, "sum[%d] got=%.17g expected=%.17g\n",
                         row, static_cast<double>(sums[row]), expected_sum);
            ok = false;
        }
        if (!std::isfinite(maxima[row]) ||
            static_cast<double>(maxima[row]) != expected_max) {
            std::fprintf(stderr, "max[%d] got=%.17g expected=%.17g\n",
                         row, static_cast<double>(maxima[row]), expected_max);
            ok = false;
        }
    }
    return ok;
}

}  // namespace

int main() {
    const int selected = select_sm75_device();
    if (selected != EXIT_SUCCESS) return selected;

    __half* a = nullptr;
    __half* b = nullptr;
    float* c = nullptr;
    float* sums = nullptr;
    float* maxima = nullptr;
    bool ok = cuda_ok(cudaMallocManaged(&a, kCells * sizeof(__half)), "cudaMallocManaged(A)");
    if (ok) ok = cuda_ok(cudaMallocManaged(&b, kCells * sizeof(__half)), "cudaMallocManaged(B)");
    if (ok) ok = cuda_ok(cudaMallocManaged(&c, kCells * sizeof(float)), "cudaMallocManaged(C)");
    if (ok) ok = cuda_ok(cudaMallocManaged(&sums, kTile * sizeof(float)), "cudaMallocManaged(sum)");
    if (ok) ok = cuda_ok(cudaMallocManaged(&maxima, kTile * sizeof(float)), "cudaMallocManaged(max)");

    if (ok) {
        for (int row = 0; row < kTile; ++row) {
            for (int col = 0; col < kTile; ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * kTile + col;
                a[index] = __float2half(static_cast<float>(numerator_a(row, col)) / 8.0f);
                b[index] = __float2half(static_cast<float>(numerator_b(row, col)) / 8.0f);
                c[index] = std::numeric_limits<float>::quiet_NaN();
            }
            sums[row] = std::numeric_limits<float>::quiet_NaN();
            maxima[row] = std::numeric_limits<float>::quiet_NaN();
        }
        gemm_row_stats<<<1, 32>>>(a, b, c, sums, maxima);
        ok = cuda_ok(cudaGetLastError(), "gemm_row_stats launch") && ok;
        ok = cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") && ok;
        if (ok) ok = check_outputs(c, sums, maxima);
    }

    if (a) ok = cuda_ok(cudaFree(a), "cudaFree(A)") && ok;
    if (b) ok = cuda_ok(cudaFree(b), "cudaFree(B)") && ok;
    if (c) ok = cuda_ok(cudaFree(c), "cudaFree(C)") && ok;
    if (sums) ok = cuda_ok(cudaFree(sums), "cudaFree(sum)") && ok;
    if (maxima) ok = cuda_ok(cudaFree(maxima), "cudaFree(max)") && ok;
    if (ok) std::printf("GEMM row statistics example: PASS\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
