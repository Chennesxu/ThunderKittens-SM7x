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

}  // namespace tk_sm7x::detail
