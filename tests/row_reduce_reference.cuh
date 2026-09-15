#pragma once

#include <cstddef>

#include "tk_sm7x/ptx_backend.cuh"

namespace tk_sm7x::test {

// The declared row-reduction order written out longhand: a balanced binary tree
// over the 16 column values in column index order. A backend row reduction must
// produce exactly this value, so this is the frozen reference rather than a
// second implementation choice.
template <class Op>
__host__ __device__ __forceinline__ float canonical_row(const float* row) {
    const float q0 = Op::apply(Op::apply(row[0], row[1]), Op::apply(row[2], row[3]));
    const float q1 = Op::apply(Op::apply(row[4], row[5]), Op::apply(row[6], row[7]));
    const float q2 = Op::apply(Op::apply(row[8], row[9]), Op::apply(row[10], row[11]));
    const float q3 = Op::apply(Op::apply(row[12], row[13]), Op::apply(row[14], row[15]));
    return Op::apply(Op::apply(q0, q1), Op::apply(q2, q3));
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
