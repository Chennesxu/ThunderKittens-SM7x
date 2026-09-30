#pragma once

#include <cmath>
#include <cstddef>

#include "tk_sm7x/row_reduce_ops.cuh"
#include "tk_sm7x/tile.cuh"

namespace tk_sm7x {

// All 32 lanes of one warp in a one-dimensional CTA with a block size multiple
// of 32 call convergently. logits is finite and unchanged. dst and scratch are
// distinct aligned shared tiles private to this warp. Each dst row contains
// exp(x - row_max) / row_sum(exp(x - row_max)), with balanced column-order
// reductions. The final full-mask warp barrier permits same-warp dst reads and
// scratch reuse on return; other warps and CTAs need caller synchronization.
__device__ __forceinline__ void row_softmax(
    st<float, 16, 16, row_major>& dst, const rt_c& logits,
    st<float, 16, 16, row_major>& scratch) {
#if defined(KITTENS_MMA_PTX)
    detail::active_warp_mma::softmax(dst.data, logits.value, scratch.data);
#else
    store(scratch, logits);
    __syncwarp(0xffffffffu);
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (lane < 16) {
        const std::size_t row_offset = static_cast<std::size_t>(lane) * 16;
        const float* row = scratch.data + row_offset;
        const float maximum = detail::balanced_row_reduce<detail::row_max_op>(row);
        float exponent[16];
#pragma unroll
        for (int col = 0; col < 16; ++col) exponent[col] = expf(row[col] - maximum);
        const float sum = detail::balanced_row_reduce<detail::row_sum_op>(exponent);
#pragma unroll
        for (int col = 0; col < 16; ++col) dst.data[row_offset + col] = exponent[col] / sum;
    }
    __syncwarp(0xffffffffu);
#endif
}

}  // namespace tk_sm7x
