#include <cuda_fp16.h>

#include <cstddef>

#include "tk_sm7x/ptx_backend.cuh"

#if defined(KITTENS_SM75)
extern "C" __global__ void codegen_ptx_backend_sm75(
    const __half* input_a, const __half* input_b, float* output) {
    __shared__ __align__(32) __half shared_a[16 * 16];
    __shared__ __align__(32) __half shared_b[16 * 16];
    __shared__ __align__(32) float shared_c[16 * 16];

    const int lane = static_cast<int>(threadIdx.x) % 32;
    for (int linear = lane; linear < 16 * 16; linear += 32) {
        const int row = linear / 16;
        const int col = linear % 16;
        shared_a[linear] = input_a[linear];
        shared_b[col * 16 + row] = input_b[linear];
    }
    __syncwarp(0xffffffffu);

    using backend =
        tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<
            tk_sm7x::arch::sm75>;
    backend::fragment_a a;
    backend::fragment_b b;
    backend::accumulator accumulator;
    backend::load_a(a, shared_a, 16);
    backend::load_b(b, shared_b, 16);
    backend::clear(accumulator);
    backend::mma(accumulator, a, b);
    backend::store(shared_c, accumulator, 16);
    __syncwarp(0xffffffffu);

    for (int linear = lane; linear < 16 * 16; linear += 32) {
        output[static_cast<std::size_t>(linear)] = shared_c[linear];
    }
}
#endif
