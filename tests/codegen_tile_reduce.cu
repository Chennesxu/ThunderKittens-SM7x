#include <cuda_fp16.h>
#include <cstddef>

#include "tk_sm7x/reduce.cuh"

extern "C" __global__ void codegen_tile_reduce(
    const __half* a, const __half* b, float* sums, float* maxima) {
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::row_major> as;
    __shared__ tk_sm7x::st<__half, 16, 16, tk_sm7x::col_major> bs;
    __shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> scratch;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    for (int linear = lane; linear < 256; linear += 32) {
        as.data[linear] = a[static_cast<std::size_t>(linear)];
        bs.data[static_cast<std::size_t>(linear % 16) * 16 + linear / 16] =
            b[static_cast<std::size_t>(linear)];
    }
    __syncwarp(0xffffffffu);
    tk_sm7x::rt_a af;
    tk_sm7x::rt_b bf;
    tk_sm7x::rt_c cf;
    tk_sm7x::load(af, as, 0);
    tk_sm7x::load(bf, bs, 0);
    tk_sm7x::zero(cf);
    tk_sm7x::mma(cf, af, bf);
    tk_sm7x::row_sum(sums, cf, scratch);
    tk_sm7x::row_max(maxima, cf, scratch);
}
