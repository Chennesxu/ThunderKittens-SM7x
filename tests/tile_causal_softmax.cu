#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "test_utils.cuh"
#include "tk_sm7x/softmax.cuh"

namespace {

constexpr int kWidth = 16;
constexpr int kCells = kWidth * kWidth;
constexpr double kBound = 512.0 * 0x1p-23;

struct Data {
    __half a[kCells];
    __half identity[kCells];
    float before[kCells];
    float after[kCells];
    float probability[kCells];
};

__global__ void tile_causal_softmax_kernel(Data* data) {
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::row_major> shared_a;
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::col_major> shared_identity;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> shared_before;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> shared_after;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> shared_probability;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> scratch;

    tk_sm7x::load_block<32>(shared_a,
        tk_sm7x::gl<const __half>{data->a, 16, 16, 16}, 0, 0);
    tk_sm7x::load_block<32>(shared_identity,
        tk_sm7x::gl<const __half>{data->identity, 16, 16, 16}, 0, 0);
    __syncthreads();

    tk_sm7x::rt_a a;
    tk_sm7x::rt_b b;
    tk_sm7x::rt_c logits;
    tk_sm7x::load(a, shared_a, 0);
    tk_sm7x::load(b, shared_identity, 0);
    tk_sm7x::zero(logits);
    tk_sm7x::mma(logits, a, b);
    tk_sm7x::store(shared_before, logits);
    __syncwarp(0xffffffffu);
    tk_sm7x::row_softmax_causal(shared_probability, logits, scratch);
    tk_sm7x::store(shared_after, logits);
    __syncwarp(0xffffffffu);

    tk_sm7x::store_block<32>(
        tk_sm7x::gl<float>{data->before, 16, 16, 16}, shared_before, 0, 0);
    tk_sm7x::store_block<32>(
        tk_sm7x::gl<float>{data->after, 16, 16, 16}, shared_after, 0, 0);
    tk_sm7x::store_block<32>(
        tk_sm7x::gl<float>{data->probability, 16, 16, 16}, shared_probability, 0, 0);
}

std::size_t cell(int row, int col) {
    return static_cast<std::size_t>(row) * kWidth + col;
}

double logit(int row, int col) {
    return static_cast<double>((3 * row + 5 * col) % 13 - 6) / 8.0;
}

bool check(const Data& data) {
    for (int row = 0; row < kWidth; ++row) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col <= row; ++col) {
            maximum = std::fmax(maximum, logit(row, col));
        }
        double denominator = 0.0;
        for (int col = 0; col <= row; ++col) {
            denominator += std::exp(logit(row, col) - maximum);
        }
        double sum = 0.0;
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t index = cell(row, col);
            const double got = static_cast<double>(data.probability[index]);
            const double want = col <= row
                ? std::exp(logit(row, col) - maximum) / denominator : 0.0;
            if (!std::isfinite(got) ||
                (col > row ? got != 0.0 : std::fabs(got - want) > kBound)) {
                std::fprintf(stderr,
                    "causal probability [%d,%d] got=%.17g want=%.17g bound=%.17g\n",
                    row, col, got, want, kBound);
                return false;
            }
            if (!std::isfinite(data.before[index]) ||
                data.before[index] != static_cast<float>(logit(row, col)) ||
                data.after[index] != data.before[index]) {
                std::fprintf(stderr, "causal source changed or unwritten at [%d,%d]\n",
                             row, col);
                return false;
            }
            sum += got;
        }
        if (std::fabs(sum - 1.0) > kBound) {
            std::fprintf(stderr, "causal row %d sum=%.17g\n", row, sum);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selected = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selected != EXIT_SUCCESS) return selected;

    Data* data = nullptr;
    if (!tk_sm7x::test::cuda_ok(cudaMallocManaged(&data, sizeof(Data)),
                                "cudaMallocManaged(data)")) return EXIT_FAILURE;
    for (int row = 0; row < kWidth; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t index = cell(row, col);
            data->a[index] = __float2half_rn(static_cast<float>(logit(row, col)));
            data->identity[index] = __float2half_rn(row == col ? 1.0f : 0.0f);
            data->before[index] = std::numeric_limits<float>::quiet_NaN();
            data->after[index] = std::numeric_limits<float>::quiet_NaN();
            data->probability[index] = std::numeric_limits<float>::quiet_NaN();
        }
    }

    tile_causal_softmax_kernel<<<1, 32>>>(data);
    bool ok = tk_sm7x::test::cuda_ok(cudaGetLastError(),
                                      "tile_causal_softmax_kernel launch");
    ok = tk_sm7x::test::cuda_ok(cudaDeviceSynchronize(),
                                "cudaDeviceSynchronize") && ok;
    if (ok) ok = check(*data);
    ok = tk_sm7x::test::cuda_ok(cudaFree(data), "cudaFree(data)") && ok;
    if (ok) std::printf("causal tile softmax: PASS ordinal=%d\n", ordinal);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
