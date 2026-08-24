#include <cuda_fp16.h>

#include "tk_sm7x/mma.cuh"

extern "C" __global__ void codegen_mma(
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

    tk_sm7x::detail::active_warp_mma::fragment_a a_fragment;
    tk_sm7x::detail::active_warp_mma::fragment_b b_fragment;
    tk_sm7x::detail::active_warp_mma::load_a(a_fragment, a_shared, 16);
    tk_sm7x::detail::active_warp_mma::load_b(b_fragment, b_shared, 16);
    tk_sm7x::detail::active_warp_mma::accumulator accumulator;
    tk_sm7x::detail::active_warp_mma::clear(accumulator);
    tk_sm7x::detail::active_warp_mma::mma(accumulator, a_fragment, b_fragment);
    tk_sm7x::detail::active_warp_mma::store(c_shared, accumulator, 16);
    __syncwarp(0xffffffffu);

    for (int linear = lane; linear < 256; linear += 32) {
        c[linear] = c_shared[linear];
    }
}
