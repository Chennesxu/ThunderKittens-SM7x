#include "tk_sm7x/gemm.cuh"

#include <cstddef>

#include "tk_sm7x/tile.cuh"

namespace tk_sm7x {
namespace {

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kWarpsM = 4;
constexpr int kWarpsN = 2;

constexpr int kWarps = kWarpsM * kWarpsN;
constexpr int kThreads = kWarps * 32;
constexpr int kFragsM = kTileM / kWarpsM / 16;
constexpr int kFragsN = kTileN / kWarpsN / 16;
constexpr unsigned int kMaxGridY = 65535u;
constexpr int kMaxExtent = 2147483632;

// Forming extent + tile - 1 overflows for extents near the top of the supported
// domain, so the rounding term is derived from the remainder instead. The
// assertions below force constant evaluation at the largest supported extent:
// an implementation that overflows there is not a constant expression and fails
// to compile.
constexpr unsigned int tile_count(int extent, int tile) {
    return static_cast<unsigned int>(extent / tile) +
           (extent % tile != 0 ? 1u : 0u);
}

static_assert(tile_count(kMaxExtent, kTileM) > 0u,
              "M tile count must not overflow at the largest supported extent");
static_assert(tile_count(kMaxExtent, kTileN) > 0u,
              "N tile count must not overflow at the largest supported extent");

__global__ __launch_bounds__(kThreads) void gemm_kernel(
    int m, int n, int k,
    const __half* a, int lda,
    const __half* b, int ldb,
    float* c, int ldc) {
    __shared__ st<__half, kTileM, 16, row_major> a_shared;
    __shared__ st<__half, 16, kTileN, col_major> b_shared;
    __shared__ st<float, 16, 16, row_major> c_stage[kWarps];

    const gl<const __half> a_global{a, m, k, lda};
    const gl<const __half> b_global{b, k, n, ldb};
    const gl<float> c_global{c, m, n, ldc};

    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;

    const std::size_t n0 = static_cast<std::size_t>(blockIdx.x) * kTileN;
    for (std::size_t m0 = static_cast<std::size_t>(blockIdx.y) * kTileM;
         m0 < static_cast<std::size_t>(m);
         m0 += static_cast<std::size_t>(gridDim.y) * kTileM) {
        rt_c accumulators[kFragsM][kFragsN];
#pragma unroll
        for (int i = 0; i < kFragsM; ++i) {
#pragma unroll
            for (int j = 0; j < kFragsN; ++j) {
                zero(accumulators[i][j]);
            }
        }

        for (std::size_t k0 = 0; k0 < static_cast<std::size_t>(k); k0 += 16) {
            load_block<kThreads>(a_shared, a_global, m0, k0);
            load_block<kThreads>(b_shared, b_global, k0, n0);
            __syncthreads();

            rt_a a_fragments[kFragsM];
            rt_b b_fragments[kFragsN];
#pragma unroll
            for (int i = 0; i < kFragsM; ++i) {
                load(a_fragments[i], a_shared, warp_m * kFragsM + i);
            }
#pragma unroll
            for (int j = 0; j < kFragsN; ++j) {
                load(b_fragments[j], b_shared, warp_n * kFragsN + j);
            }
#pragma unroll
            for (int i = 0; i < kFragsM; ++i) {
#pragma unroll
                for (int j = 0; j < kFragsN; ++j) {
                    mma(accumulators[i][j], a_fragments[i], b_fragments[j]);
                }
            }
            __syncthreads();
        }

        for (int i = 0; i < kFragsM; ++i) {
            for (int j = 0; j < kFragsN; ++j) {
                store(c_stage[warp], accumulators[i][j]);
                __syncwarp(0xffffffffu);
                const std::size_t row =
                    m0 + static_cast<std::size_t>((warp_m * kFragsM + i) * 16);
                const std::size_t col =
                    n0 + static_cast<std::size_t>((warp_n * kFragsN + j) * 16);
                store_warp(c_global, c_stage[warp], row, col);
                __syncwarp(0xffffffffu);
            }
        }
        __syncthreads();
    }
}

}  // namespace

cudaError_t gemm_f16_f16_f32_nn(
    int m, int n, int k,
    const __half* a, int lda,
    const __half* b, int ldb,
    float* c, int ldc,
    cudaStream_t stream) {
    if (a == nullptr || b == nullptr || c == nullptr ||
        m <= 0 || n <= 0 || k <= 0 ||
        m % 16 != 0 || n % 16 != 0 || k % 16 != 0 ||
        lda < k || ldb < n || ldc < n) {
        return cudaErrorInvalidValue;
    }

    const unsigned int m_tiles = tile_count(m, kTileM);
    const dim3 grid(tile_count(n, kTileN),
                    m_tiles < kMaxGridY ? m_tiles : kMaxGridY);
    static_cast<void>(cudaGetLastError());
    gemm_kernel<<<grid, kThreads, 0, stream>>>(m, n, k, a, lda, b, ldb, c, ldc);
    return cudaGetLastError();
}

}  // namespace tk_sm7x
