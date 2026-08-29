#include "tk_sm7x/gemm.cuh"

#include <cstddef>

#include "tk_sm7x/tile.cuh"

namespace tk_sm7x {
namespace {

constexpr int kLargeTileM = 128;
constexpr int kLargeTileN = 128;
constexpr int kLargeWarpsM = 4;
constexpr int kLargeWarpsN = 2;

constexpr int kSmallTileM = 64;
constexpr int kSmallTileN = 64;
constexpr int kSmallWarpsM = 2;
constexpr int kSmallWarpsN = 2;

// Below this many large-tile CTAs the grid is too small to pay for the large
// tile's register pressure, and the smaller tile wins despite its lower
// arithmetic intensity. Measured on 72 SMs: the small tile leads up to 56 CTAs
// and the large tile leads from 64, across square, rectangular and K-varying
// shapes alike.
constexpr std::size_t kSmallTileCtaThreshold = 64u;

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

static_assert(tile_count(kMaxExtent, kSmallTileM) > 0u &&
                  tile_count(kMaxExtent, kLargeTileM) > 0u,
              "M tile count must not overflow at the largest supported extent");
static_assert(tile_count(kMaxExtent, kSmallTileN) > 0u &&
                  tile_count(kMaxExtent, kLargeTileN) > 0u,
              "N tile count must not overflow at the largest supported extent");
// The single definition of tile selection. The launcher and the assertions below
// share it so a threshold change cannot silently route every correctness case
// through one kernel.
constexpr std::size_t large_cta_count(int m, int n) {
    return static_cast<std::size_t>(tile_count(m, kLargeTileM)) *
           static_cast<std::size_t>(tile_count(n, kLargeTileN));
}

constexpr bool use_small_tile(int m, int n) {
    return large_cta_count(m, n) < kSmallTileCtaThreshold;
}

static_assert(use_small_tile(256, 384),
              "small-tile correctness case must exercise the small kernel");
static_assert(!use_small_tile(1024, 1024),
              "large-tile correctness case must exercise the large kernel");
static_assert(!use_small_tile(8388608, 16),
              "grid-y boundary case must exercise the large kernel");

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ __launch_bounds__(WarpsM * WarpsN * 32) void gemm_kernel(
    int m, int n, int k,
    const __half* a, int lda,
    const __half* b, int ldb,
    float* c, int ldc) {
    constexpr int kWarps = WarpsM * WarpsN;
    constexpr int kThreads = kWarps * 32;
    constexpr int kFragsM = TileM / WarpsM / 16;
    constexpr int kFragsN = TileN / WarpsN / 16;

    __shared__ st<__half, TileM, 16, row_major> a_shared;
    __shared__ st<__half, 16, TileN, col_major> b_shared;
    __shared__ st<float, 16, 16, row_major> c_stage[kWarps];

    const gl<const __half> a_global{a, m, k, lda};
    const gl<const __half> b_global{b, k, n, ldb};
    const gl<float> c_global{c, m, n, ldc};

    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int warp_m = warp / WarpsN;
    const int warp_n = warp % WarpsN;

    const std::size_t n0 = static_cast<std::size_t>(blockIdx.x) * TileN;
    for (std::size_t m0 = static_cast<std::size_t>(blockIdx.y) * TileM;
         m0 < static_cast<std::size_t>(m);
         m0 += static_cast<std::size_t>(gridDim.y) * TileM) {
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

template <int TileM, int TileN, int WarpsM, int WarpsN>
void launch(
    int m, int n, int k,
    const __half* a, int lda,
    const __half* b, int ldb,
    float* c, int ldc,
    cudaStream_t stream) {
    const unsigned int m_tiles = tile_count(m, TileM);
    const dim3 grid(tile_count(n, TileN),
                    m_tiles < kMaxGridY ? m_tiles : kMaxGridY);
    gemm_kernel<TileM, TileN, WarpsM, WarpsN>
        <<<grid, WarpsM * WarpsN * 32, 0, stream>>>(m, n, k, a, lda, b, ldb, c, ldc);
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

    static_cast<void>(cudaGetLastError());
    if (use_small_tile(m, n)) {
        launch<kSmallTileM, kSmallTileN, kSmallWarpsM, kSmallWarpsN>(
            m, n, k, a, lda, b, ldb, c, ldc, stream);
    } else {
        launch<kLargeTileM, kLargeTileN, kLargeWarpsM, kLargeWarpsN>(
            m, n, k, a, lda, b, ldb, c, ldc, stream);
    }
    return cudaGetLastError();
}

}  // namespace tk_sm7x
