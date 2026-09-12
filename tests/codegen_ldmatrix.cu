#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "tk_sm7x/ptx_ldmatrix.cuh"

extern "C" __global__ void codegen_ldmatrix_x2(const __half* input, uint32_t* output) {
    __shared__ __align__(32) __half shared[16 * 8];
    const int lane = static_cast<int>(threadIdx.x);
    for (int linear = lane; linear < 16 * 8; linear += 32) {
        shared[linear] = input[linear];
    }
    __syncthreads();

    uint32_t loaded[2];
    const int owner = lane % 16;
    tk_sm7x::detail::ldmatrix_x2(
        loaded, tk_sm7x::detail::shared_address(shared + owner * 8));
    const std::size_t output_base = static_cast<std::size_t>(lane) * 2;
    output[output_base] = loaded[0];
    output[output_base + 1] = loaded[1];
}

extern "C" __global__ void codegen_ldmatrix_x1(const __half* input, uint32_t* output) {
    __shared__ __align__(32) __half shared[8 * 8];
    const int lane = static_cast<int>(threadIdx.x);
    for (int linear = lane; linear < 8 * 8; linear += 32) {
        shared[linear] = input[linear];
    }
    __syncthreads();

    uint32_t loaded;
    const int owner = lane % 8;
    tk_sm7x::detail::ldmatrix_x1(
        loaded, tk_sm7x::detail::shared_address(shared + owner * 8));
    output[static_cast<std::size_t>(lane)] = loaded;
}
