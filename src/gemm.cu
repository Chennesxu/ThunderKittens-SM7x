#include "tk_sm7x/gemm.cuh"

#include <cstddef>

#include "tk_sm7x/tile.cuh"

namespace tk_sm7x {
namespace {

__global__ void gemm_kernel(
    int m, int k,
    const __half* a, int lda,
    const __half* b, int ldb,
    float* c, int ldc) {
    __shared__ st<__half, 16, 16, row_major> a_shared;
    __shared__ st<__half, 16, 16, col_major> b_shared;
    __shared__ st<float, 16, 16, row_major> c_shared;

    const gl<const __half> a_global{a, lda};
    const gl<const __half> b_global{b, ldb};
    const gl<float> c_global{c, ldc};

    const std::size_t n0 = static_cast<std::size_t>(blockIdx.x) * 16;
    for (std::size_t m0 = static_cast<std::size_t>(blockIdx.y) * 16;
         m0 < static_cast<std::size_t>(m);
         m0 += static_cast<std::size_t>(gridDim.y) * 16) {
        rt_c accumulator;
        zero(accumulator);

        for (std::size_t k0 = 0; k0 < static_cast<std::size_t>(k); k0 += 16) {
            load(a_shared, a_global, m0, k0);
            load(b_shared, b_global, k0, n0);
            __syncwarp(0xffffffffu);
            rt_a a_fragment;
            rt_b b_fragment;
            load(a_fragment, a_shared);
            load(b_fragment, b_shared);
            mma(accumulator, a_fragment, b_fragment);
            __syncwarp(0xffffffffu);
        }

        store(c_shared, accumulator);
        __syncwarp(0xffffffffu);
        store(c_global, c_shared, m0, n0);
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

    const unsigned int m_tiles = static_cast<unsigned int>(m / 16);
    const dim3 grid(static_cast<unsigned int>(n / 16),
                    m_tiles < 65535u ? m_tiles : 65535u);
    static_cast<void>(cudaGetLastError());
    gemm_kernel<<<grid, 32, 0, stream>>>(m, k, a, lda, b, ldb, c, ldc);
    return cudaGetLastError();
}

}  // namespace tk_sm7x
