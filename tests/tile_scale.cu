#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "test_utils.cuh"
#include "tk_sm7x/tile.cuh"

namespace {

constexpr int kWidth = 16;
constexpr int kCells = kWidth * kWidth;

struct Data {
    __half a[kCells];
    __half b[kCells];
    float result[2][kCells];
};

__global__ void tile_scale_kernel(const __half* a, const __half* b,
                                  float* result, float factor) {
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::row_major> shared_a;
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::col_major> shared_b;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> shared_c;

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
    tk_sm7x::scale(c_tile, factor);
    tk_sm7x::store(shared_c, c_tile);
    __syncwarp(0xffffffffu);
    tk_sm7x::store_block<32>(tk_sm7x::gl<float>{result, 16, 16, 16}, shared_c, 0, 0);
}

bool check_result(const Data& data, int case_index, float factor) {
    for (int row = 0; row < kWidth; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * kWidth + col;
            const float got = data.result[case_index][index];
            const float want = factor * (static_cast<float>(row + 1) / 16.0f) *
                               (static_cast<float>(16 * row + col - 128) / 8.0f);
            if (!std::isfinite(got) || got != want) {
                std::fprintf(stderr, "scale factor=%g row=%d col=%d got=%g want=%g\n",
                             static_cast<double>(factor), row, col,
                             static_cast<double>(got), static_cast<double>(want));
                return false;
            }
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
            const std::size_t index = static_cast<std::size_t>(row) * kWidth + col;
            data->a[index] = __float2half_rn(row == col
                ? static_cast<float>(row + 1) / 16.0f : 0.0f);
            data->b[index] = __float2half_rn(
                static_cast<float>(16 * row + col - 128) / 8.0f);
            data->result[0][index] = std::numeric_limits<float>::quiet_NaN();
            data->result[1][index] = std::numeric_limits<float>::quiet_NaN();
        }
    }

    constexpr float factors[2] = {0.25f, -0.5f};
    bool ok = true;
    for (int i = 0; i < 2; ++i) {
        tile_scale_kernel<<<1, 32>>>(data->a, data->b, data->result[i], factors[i]);
        ok = tk_sm7x::test::cuda_ok(cudaGetLastError(), "tile_scale_kernel launch") && ok;
    }
    ok = tk_sm7x::test::cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") && ok;
    if (ok) {
        for (int i = 0; i < 2; ++i) ok = check_result(*data, i, factors[i]) && ok;
    }
    ok = tk_sm7x::test::cuda_ok(cudaFree(data), "cudaFree(data)") && ok;
    if (ok) std::printf("tile scale: PASS ordinal=%d\n", ordinal);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
