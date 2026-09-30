#pragma once

namespace tk_sm7x::detail {

struct row_sum_op {
    __host__ __device__ static __forceinline__ float apply(float low, float high) {
        return low + high;
    }
};

struct row_max_op {
    __host__ __device__ static __forceinline__ float apply(float low, float high) {
        return low > high ? low : high;
    }
};

template <class Op>
__device__ __forceinline__ float balanced_row_reduce(const float* row) {
    const float q0 = Op::apply(Op::apply(row[0], row[1]), Op::apply(row[2], row[3]));
    const float q1 = Op::apply(Op::apply(row[4], row[5]), Op::apply(row[6], row[7]));
    const float q2 = Op::apply(Op::apply(row[8], row[9]), Op::apply(row[10], row[11]));
    const float q3 = Op::apply(Op::apply(row[12], row[13]), Op::apply(row[14], row[15]));
    return Op::apply(Op::apply(q0, q1), Op::apply(q2, q3));
}

}  // namespace tk_sm7x::detail
