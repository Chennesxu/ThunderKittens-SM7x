#include <cstddef>

#include "tk_sm7x/ptx_backend.cuh"

static_assert(sizeof(tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<
                      tk_sm7x::arch::sm75>) > 0,
              "forced backend instantiation must be complete");
