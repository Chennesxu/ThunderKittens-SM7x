#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "tk_sm7x/arch.cuh"

namespace tk_sm7x::detail {

// Raw MMA instructions and the register layouts they impose. Every operation is
// warp-collective: all 32 lanes must reach it converged. Hardware measurements
// establish the lane-to-matrix and accumulator mappings. operand_slot freezes a
// software K-slot convention whose absolute hardware order is a separate staging
// contract; tests/mma_layout.cu pins both kinds of invariant.
//
// The instruction operands are packed 32-bit registers, and that is what these
// wrappers accept. A caller must not reinterpret a __half array as uint32_t:
// the array carries only 2-byte alignment and the access would also break type
// aliasing. pack_halves goes through the half bit-pattern intrinsic instead.

__device__ __forceinline__ uint32_t pack_halves(__half low, __half high) {
    return static_cast<uint32_t>(__half_as_ushort(low)) |
           (static_cast<uint32_t>(__half_as_ushort(high)) << 16);
}

// mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32, available on SM70 and SM75.
// One instruction performs four independent 8x8x4 products, one per quadpair.
// Quadpair q owns lanes {4q..4q+3} and {16+4q..16+4q+3}; within it a lane holds
// row and column `position` of the operands and four consecutive k elements.
struct mma_m8n8k4 {
    static constexpr int kQuadpairs = 4;
    static constexpr int kTile = 8;
    static constexpr int kDepth = 4;
    static constexpr int kAccumulators = 8;
    static constexpr int kOperandRegisters = 2;

    // Which element of a lane's operand run occupies the low (half 0) or high
    // (half 1) part of a packed register. A caller must place operands through
    // this rather than choosing a pack order of its own: the contraction is
    // invariant under a k permutation applied to both operands, so a divergent
    // order stays numerically silent until an operand arrives from hardware.
    __host__ __device__ static __forceinline__ int operand_slot(int reg, int half) {
        return 2 * reg + half;
    }

    __host__ __device__ static __forceinline__ int quadpair(int lane) {
        return (lane % 16) / 4;
    }

    __host__ __device__ static __forceinline__ int position(int lane) {
        return (lane % 4) + 4 * (lane / 16);
    }

    __host__ __device__ static __forceinline__ int accumulator_lane(
        int quadpair, int row, int col) {
        return 4 * quadpair + (row % 2) + 2 * ((col % 4) / 2) + 16 * (row / 4);
    }

    __host__ __device__ static __forceinline__ int accumulator_index(int row, int col) {
        return (col % 2) + 2 * ((row % 4) / 2) + 4 * (col / 4);
    }

    __device__ static __forceinline__ void mma(
        float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
            "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
              "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1),
              "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
              "f"(d[4]), "f"(d[5]), "f"(d[6]), "f"(d[7]));
    }
};

#if defined(KITTENS_SM75)
// mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32, SM75 and later only.
// One instruction performs a single 16x8x8 product across the whole warp.
// group = lane / 4 indexes rows and B columns; lane % 4 indexes column pairs.
struct mma_m16n8k8 {
    static constexpr int kRows = 16;
    static constexpr int kCols = 8;
    static constexpr int kDepth = 8;
    static constexpr int kAccumulators = 4;
    static constexpr int kOperandARegisters = 2;
    static constexpr int kOperandBRegisters = 1;

    __host__ __device__ static __forceinline__ int operand_slot(int reg, int half) {
        return 2 * reg + half;
    }

    __host__ __device__ static __forceinline__ int group(int lane) { return lane / 4; }

    __host__ __device__ static __forceinline__ int pair(int lane) { return lane % 4; }

    __host__ __device__ static __forceinline__ int operand_a_row(int lane, int index) {
        return group(lane) + 8 * (index / 2);
    }

    __host__ __device__ static __forceinline__ int operand_a_col(int lane, int index) {
        return pair(lane) * 2 + index % 2;
    }

    __host__ __device__ static __forceinline__ int operand_b_row(int lane, int index) {
        return pair(lane) * 2 + index;
    }

    __host__ __device__ static __forceinline__ int operand_b_col(int lane) {
        return group(lane);
    }

    __host__ __device__ static __forceinline__ int accumulator_row(int lane, int index) {
        return group(lane) + 8 * (index / 2);
    }

    __host__ __device__ static __forceinline__ int accumulator_col(int lane, int index) {
        return pair(lane) * 2 + index % 2;
    }

    __device__ static __forceinline__ void mma(
        float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a0), "r"(a1), "r"(b0),
              "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
    }
};
#endif

}  // namespace tk_sm7x::detail
