#include "tk_sm7x/gemm.cuh"

#include "tk_sm7x/mma.cuh"

namespace tk_sm7x {
namespace {

__global__ void compact_gemm_kernel(
    const __half* a, const __half* b, float* c) {
    __shared__ __align__(32) __half a_shared[256];
    __shared__ __align__(32) __half b_shared[256];
    __shared__ __align__(32) float c_shared[256];

    const int lane = static_cast<int>(threadIdx.x);
    for (int linear = lane; linear < 256; linear += 32) {
        const int row = linear / 16;
        const int col = linear % 16;
        a_shared[linear] = a[linear];
        b_shared[col * 16 + row] = b[linear];
    }
    __syncwarp(0xffffffffu);

    detail::active_warp_mma::accumulator accumulator;
    detail::active_warp_mma::clear(accumulator);
    detail::active_warp_mma::mma(accumulator, a_shared, b_shared);
    detail::active_warp_mma::store(c_shared, accumulator);
    __syncwarp(0xffffffffu);

    for (int linear = lane; linear < 256; linear += 32) {
        c[linear] = c_shared[linear];
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
        m != 16 || n != 16 || k != 16 ||
        lda != 16 || ldb != 16 || ldc != 16) {
        return cudaErrorInvalidValue;
    }

    static_cast<void>(cudaGetLastError());
    compact_gemm_kernel<<<1, 32, 0, stream>>>(a, b, c);
    return cudaGetLastError();
}

}  // namespace tk_sm7x
