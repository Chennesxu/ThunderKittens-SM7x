#include <cuda_fp16.h>

#include <cstddef>

#include "tk_sm7x/ptx_backend.cuh"

template <class Backend>
__device__ __forceinline__ void codegen_ptx_backend_probe(
    const __half* input_a, const __half* input_b, float* output,
    __half* shared_a, __half* shared_b, float* shared_c) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    for (int linear = lane; linear < 16 * 16; linear += 32) {
        const int row = linear / 16;
        const int col = linear % 16;
        shared_a[linear] = input_a[linear];
        shared_b[col * 16 + row] = input_b[linear];
    }
    __syncwarp(0xffffffffu);

    typename Backend::fragment_a a;
    typename Backend::fragment_b b;
    typename Backend::accumulator accumulator;
    Backend::load_a(a, shared_a, 16);
    Backend::load_b(b, shared_b, 16);
    Backend::clear(accumulator);
    Backend::mma(accumulator, a, b);
    Backend::store(shared_c, accumulator, 16);
    __syncwarp(0xffffffffu);

    for (int linear = lane; linear < 16 * 16; linear += 32) {
        output[static_cast<std::size_t>(linear)] = shared_c[linear];
    }
}

extern "C" __global__ void codegen_ptx_backend_sm70(
    const __half* input_a, const __half* input_b, float* output) {
    __shared__ __align__(32) __half shared_a[16 * 16];
    __shared__ __align__(32) __half shared_b[16 * 16];
    __shared__ __align__(32) float shared_c[16 * 16];
    using backend =
        tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<
            tk_sm7x::arch::sm70>;
    codegen_ptx_backend_probe<backend>(input_a, input_b, output, shared_a,
                                       shared_b, shared_c);
}

#if defined(KITTENS_SM75)
extern "C" __global__ void codegen_ptx_backend_sm75(
    const __half* input_a, const __half* input_b, float* output) {
    __shared__ __align__(32) __half shared_a[16 * 16];
    __shared__ __align__(32) __half shared_b[16 * 16];
    __shared__ __align__(32) float shared_c[16 * 16];
    using backend =
        tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<
            tk_sm7x::arch::sm75>;
    codegen_ptx_backend_probe<backend>(input_a, input_b, output, shared_a,
                                       shared_b, shared_c);
}
#endif
