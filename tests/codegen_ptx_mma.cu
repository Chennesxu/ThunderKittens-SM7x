#include <cuda_fp16.h>

#include <cstdint>

#include "tk_sm7x/ptx_mma.cuh"

extern "C" __global__ void codegen_ptx_m8n8k4(
    const __half* a, const __half* b, float* c) {
    float accumulator[8];
    for (int i = 0; i < 8; ++i) {
        accumulator[i] = c[i];
    }
    tk_sm7x::detail::mma_m8n8k4::mma(
        accumulator,
        tk_sm7x::detail::pack_halves(a[0], a[1]),
        tk_sm7x::detail::pack_halves(a[2], a[3]),
        tk_sm7x::detail::pack_halves(b[0], b[1]),
        tk_sm7x::detail::pack_halves(b[2], b[3]));
    for (int i = 0; i < 8; ++i) {
        c[i] = accumulator[i];
    }
}

#if defined(KITTENS_SM75)
extern "C" __global__ void codegen_ptx_m16n8k8(
    const __half* a, const __half* b, float* c) {
    float accumulator[4];
    for (int i = 0; i < 4; ++i) {
        accumulator[i] = c[i];
    }
    tk_sm7x::detail::mma_m16n8k8::mma(
        accumulator,
        tk_sm7x::detail::pack_halves(a[0], a[1]),
        tk_sm7x::detail::pack_halves(a[2], a[3]),
        tk_sm7x::detail::pack_halves(b[0], b[1]));
    for (int i = 0; i < 4; ++i) {
        c[i] = accumulator[i];
    }
}
#endif
