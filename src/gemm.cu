#include "tk_sm7x/gemm.cuh"

#include <cstddef>

#include "tk_sm7x/mma.cuh"

namespace tk_sm7x {
namespace {

__global__ void gemm_kernel(
    int k,
    const __half* a, int lda,
    const __half* b, int ldb,
    float* c, int ldc) {
    __shared__ __align__(32) __half a_shared[256];
    __shared__ __align__(32) __half b_shared[256];
    __shared__ __align__(32) float c_shared[256];

    const int lane = static_cast<int>(threadIdx.x);
    const int m0 = static_cast<int>(blockIdx.y) * 16;
    const int n0 = static_cast<int>(blockIdx.x) * 16;
    detail::active_warp_mma::accumulator accumulator;
    detail::active_warp_mma::clear(accumulator);

    for (int k0 = 0; k0 < k; k0 += 16) {
        for (int linear = lane; linear < 256; linear += 32) {
            const int row = linear / 16;
            const int col = linear % 16;
            const std::size_t a_index =
                static_cast<std::size_t>(m0 + row) * static_cast<std::size_t>(lda) +
                static_cast<std::size_t>(k0 + col);
            const std::size_t b_index =
                static_cast<std::size_t>(k0 + row) * static_cast<std::size_t>(ldb) +
                static_cast<std::size_t>(n0 + col);
            a_shared[linear] = a[a_index];
            b_shared[col * 16 + row] = b[b_index];
        }
        __syncwarp(0xffffffffu);
        detail::active_warp_mma::mma(accumulator, a_shared, b_shared);
        __syncwarp(0xffffffffu);
    }

    detail::active_warp_mma::store(c_shared, accumulator);
    __syncwarp(0xffffffffu);
    for (int linear = lane; linear < 256; linear += 32) {
        const int row = linear / 16;
        const int col = linear % 16;
        const std::size_t c_index =
            static_cast<std::size_t>(m0 + row) * static_cast<std::size_t>(ldc) +
            static_cast<std::size_t>(n0 + col);
        c[c_index] = c_shared[linear];
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

    const dim3 grid(static_cast<unsigned int>(n / 16),
                    static_cast<unsigned int>(m / 16));
    static_cast<void>(cudaGetLastError());
    gemm_kernel<<<grid, 32, 0, stream>>>(k, a, lda, b, ldb, c, ldc);
    return cudaGetLastError();
}

}  // namespace tk_sm7x
