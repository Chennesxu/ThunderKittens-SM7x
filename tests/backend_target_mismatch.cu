#include "tk_sm7x/mma.cuh"

#if defined(KITTENS_SM70)
using wrong_arch = tk_sm7x::arch::sm75;
#else
using wrong_arch = tk_sm7x::arch::sm70;
#endif

__global__ void instantiate_wrong_backend() {
    tk_sm7x::detail::warp_mma_f16_f16_f32_16x16x16<wrong_arch>::accumulator value;
    static_cast<void>(value);
}
