#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "tk_sm7x/softmax.cuh"

namespace {

constexpr int kTile = 16;
constexpr int kCells = kTile * kTile;
constexpr double kBound = 512.0 * 0x1p-23;

struct Data {
    __half a[kCells];
    __half b[kCells];
    float logits[kCells];
    float probabilities[kCells];
};

__global__ void gemm_row_softmax_example(Data* data) {
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::row_major> shared_a;
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::col_major> shared_b;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> shared_logits;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> probabilities;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> scratch;

    tk_sm7x::load_block<32>(shared_a, tk_sm7x::gl<const __half>{data->a, 16, 16, 16}, 0, 0);
    tk_sm7x::load_block<32>(shared_b, tk_sm7x::gl<const __half>{data->b, 16, 16, 16}, 0, 0);
    __syncthreads();

    tk_sm7x::rt_a a_tile;
    tk_sm7x::rt_b b_tile;
    tk_sm7x::rt_c c_tile;
    tk_sm7x::load(a_tile, shared_a, 0);
    tk_sm7x::load(b_tile, shared_b, 0);
    tk_sm7x::zero(c_tile);
    tk_sm7x::mma(c_tile, a_tile, b_tile);
    tk_sm7x::store(shared_logits, c_tile);
    __syncwarp();
    tk_sm7x::store_block<32>(tk_sm7x::gl<float>{data->logits, 16, 16, 16}, shared_logits, 0, 0);
    tk_sm7x::row_softmax(probabilities, c_tile, scratch);
    tk_sm7x::store_block<32>(tk_sm7x::gl<float>{data->probabilities, 16, 16, 16}, probabilities, 0, 0);
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

bool check_outputs(const Data& data) {
    bool ok = true;
    double max_error = 0.0;
    for (int row = 0; row < kTile; ++row) {
        double logits[kTile];
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < kTile; ++col) {
            double expected = 0.0;
            for (int k = 0; k < kTile; ++k) {
                expected += static_cast<double>(numerator_a(row, k)) *
                            static_cast<double>(numerator_b(k, col)) / 64.0;
            }
            logits[col] = expected;
            if (expected > maximum) maximum = expected;
            const std::size_t index = static_cast<std::size_t>(row) * kTile + col;
            if (!std::isfinite(data.logits[index]) ||
                static_cast<double>(data.logits[index]) != expected) {
                std::fprintf(stderr, "logits[%d,%d] got=%.17g expected=%.17g\n",
                             row, col, static_cast<double>(data.logits[index]), expected);
                ok = false;
            }
        }
        double exponent[kTile];
        double denominator = 0.0;
        for (int col = 0; col < kTile; ++col) {
            exponent[col] = std::exp(logits[col] - maximum);
            denominator += exponent[col];
        }
        double row_sum = 0.0;
        for (int col = 0; col < kTile; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * kTile + col;
            const double got = static_cast<double>(data.probabilities[index]);
            const double expected = exponent[col] / denominator;
            const double error = std::fabs(got - expected);
            if (error > max_error) max_error = error;
            if (!std::isfinite(got) || got < 0.0 || got > 1.0 || error > kBound) {
                std::fprintf(stderr, "probabilities[%d,%d] got=%.17g expected=%.17g\n",
                             row, col, got, expected);
                ok = false;
            }
            row_sum += got;
        }
        if (!std::isfinite(row_sum) || std::fabs(row_sum - 1.0) > kBound) {
            std::fprintf(stderr, "row %d sum=%.17g expected=1\n", row, row_sum);
            ok = false;
        }
    }
    if (ok) std::printf("GEMM row softmax example: PASS max-error=%.17g bound=%.17g\n",
                        max_error, kBound);
    return ok;
}

}  // namespace

int main() {
    const int selected = select_sm75_device();
    if (selected != EXIT_SUCCESS) return selected;

    Data* data = nullptr;
    bool ok = cuda_ok(cudaMallocManaged(&data, sizeof(Data)), "cudaMallocManaged(data)");
    if (ok) {
        for (int row = 0; row < kTile; ++row) {
            for (int col = 0; col < kTile; ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * kTile + col;
                data->a[index] = __float2half(static_cast<float>(numerator_a(row, col)) / 8.0f);
                data->b[index] = __float2half(static_cast<float>(numerator_b(row, col)) / 8.0f);
                data->logits[index] = std::numeric_limits<float>::quiet_NaN();
                data->probabilities[index] = std::numeric_limits<float>::quiet_NaN();
            }
        }
        gemm_row_softmax_example<<<1, 32>>>(data);
        ok = cuda_ok(cudaGetLastError(), "gemm_row_softmax_example launch") && ok;
        ok = cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") && ok;
        if (ok) ok = check_outputs(*data);
    }
    if (data) ok = cuda_ok(cudaFree(data), "cudaFree(data)") && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
