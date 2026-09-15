#pragma once

#include <cstddef>

#include "tk_sm7x/ptx_backend.cuh"

namespace tk_sm7x::test {

constexpr int kCanonicalRowAdditions = 15;

// The declared row-reduction order written out longhand: a balanced binary tree
// over the 16 column values in column index order. A backend row reduction must
// produce exactly this value, so this is the frozen reference rather than a
// second implementation choice. Every consumer walks this one structure, so a
// check on the order cannot drift from the reference it checks.
template <class Combine>
__host__ __device__ __forceinline__ float canonical_row_tree(
    Combine& combine, const float* row) {
    const float q0 = combine(combine(row[0], row[1]), combine(row[2], row[3]));
    const float q1 = combine(combine(row[4], row[5]), combine(row[6], row[7]));
    const float q2 = combine(combine(row[8], row[9]), combine(row[10], row[11]));
    const float q3 = combine(combine(row[12], row[13]), combine(row[14], row[15]));
    return combine(combine(q0, q1), combine(q2, q3));
}

template <class Op>
struct apply_combine {
    __host__ __device__ __forceinline__ float operator()(float low, float high) const {
        return Op::apply(low, high);
    }
};

// Knuth 2Sum: for finite operands that do not overflow, low + high equals
// sum + error exactly, so a nonzero error is proof that this addition rounded.
// Counting them over the tree is what shows the reduction itself rounds; a
// deviation from the ideal product sum would also count the rounding of the MMA
// that produced the operands.
struct rounding_census_combine {
    int inexact = 0;

    __host__ __device__ __forceinline__ float operator()(float low, float high) {
        const float sum = low + high;
        const float shifted = sum - low;
        const float error = (low - (sum - shifted)) + (high - shifted);
        if (error != 0.0f) ++inexact;
        return sum;
    }
};

template <class Op>
__host__ __device__ __forceinline__ float canonical_row(const float* row) {
    apply_combine<Op> combine;
    return canonical_row_tree(combine, row);
}

// Shared-memory round-trip reduction: the path an opaque fragment layout forces.
// Warp-collective; the caller synchronizes the staged tile before the call and
// before reading the destination. Lanes 0..15 each own one row.
template <class Op>
__device__ __forceinline__ void reference_row_reduce(
    float* destination, const float* tile) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (lane < 16) {
        destination[lane] = canonical_row<Op>(tile + static_cast<std::size_t>(lane) * 16);
    }
}

}  // namespace tk_sm7x::test
