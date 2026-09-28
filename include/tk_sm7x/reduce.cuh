#pragma once

#include <cstddef>
#include <type_traits>

#include "tk_sm7x/row_reduce_ops.cuh"
#include "tk_sm7x/tile.cuh"

namespace tk_sm7x {
namespace detail {

// All lanes of one warp in a one-dimensional CTA whose block size is a multiple
// of 32 call convergently. dst points to 16 contiguous floats in shared or
// global memory, is uniform across lanes, and does not overlap scratch.
// Concurrent warps need disjoint destinations and private shared scratch. The
// final warp barrier makes output reads and scratch reuse safe on return;
// cross-warp and cross-CTA consumers must synchronize separately. Finite values
// and finite sum intermediates are required. Columns combine in balanced index
// order; maximum ties select the right operand.
template <class Op>
__device__ __forceinline__ void reduce_rows(
    float* dst, const rt_c& src, st<float, 16, 16, row_major>& scratch) {
#if defined(KITTENS_MMA_PTX)
    if constexpr (std::is_same<Op, detail::row_sum_op>::value) {
        detail::active_warp_mma::row_sum(dst, src.value);
    } else {
        detail::active_warp_mma::row_max(dst, src.value);
    }
#else
    store(scratch, src);
    __syncwarp(0xffffffffu);
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (lane < 16) {
        const float* row = scratch.data + static_cast<std::size_t>(lane) * 16;
        const float q0 = Op::apply(Op::apply(row[0], row[1]), Op::apply(row[2], row[3]));
        const float q1 = Op::apply(Op::apply(row[4], row[5]), Op::apply(row[6], row[7]));
        const float q2 = Op::apply(Op::apply(row[8], row[9]), Op::apply(row[10], row[11]));
        const float q3 = Op::apply(Op::apply(row[12], row[13]), Op::apply(row[14], row[15]));
        dst[lane] = Op::apply(Op::apply(q0, q1), Op::apply(q2, q3));
    }
#endif
    __syncwarp(0xffffffffu);
}

}  // namespace detail

__device__ __forceinline__ void row_sum(
    float* dst, const rt_c& src, st<float, 16, 16, row_major>& scratch) {
    detail::reduce_rows<detail::row_sum_op>(dst, src, scratch);
}

__device__ __forceinline__ void row_max(
    float* dst, const rt_c& src, st<float, 16, 16, row_major>& scratch) {
    detail::reduce_rows<detail::row_max_op>(dst, src, scratch);
}

}  // namespace tk_sm7x
