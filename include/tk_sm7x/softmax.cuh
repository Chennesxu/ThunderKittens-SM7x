#pragma once

#include <cmath>
#include <cstddef>

#include "tk_sm7x/row_reduce_ops.cuh"
#include "tk_sm7x/tile.cuh"

namespace tk_sm7x {
namespace detail {

template <class Op>
__device__ __forceinline__ float balanced_half_reduce(const float* values) {
    const float low = Op::apply(Op::apply(values[0], values[1]),
                                Op::apply(values[2], values[3]));
    const float high = Op::apply(Op::apply(values[4], values[5]),
                                 Op::apply(values[6], values[7]));
    return Op::apply(low, high);
}

__device__ __forceinline__ void shared_softmax_two_lane(float* destination,
                                                        const float* source) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int row = lane / 2;
    const int half = lane % 2;
    const std::size_t offset = static_cast<std::size_t>(row) * 16 + half * 8;
    const float* values = source + offset;
    const float half_max = balanced_half_reduce<row_max_op>(values);
    const float other_max = __shfl_xor_sync(0xffffffffu, half_max, 1);
    const float maximum = half == 0
        ? row_max_op::apply(half_max, other_max)
        : row_max_op::apply(other_max, half_max);
    float exponent[8];
#pragma unroll
    for (int col = 0; col < 8; ++col) exponent[col] = expf(values[col] - maximum);
    const float half_sum = balanced_half_reduce<row_sum_op>(exponent);
    const float other_sum = __shfl_xor_sync(0xffffffffu, half_sum, 1);
    const float sum = half == 0 ? half_sum + other_sum : other_sum + half_sum;
#pragma unroll
    for (int col = 0; col < 8; ++col) {
        destination[offset + col] = exponent[col] / sum;
    }
    __syncwarp(0xffffffffu);
}

}  // namespace detail

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
    detail::shared_softmax_two_lane(dst.data, scratch.data);
#endif
}

}  // namespace tk_sm7x
