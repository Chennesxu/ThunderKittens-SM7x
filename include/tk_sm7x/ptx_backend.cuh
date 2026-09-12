#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "tk_sm7x/arch.cuh"
#include "tk_sm7x/ptx_ldmatrix.cuh"
#include "tk_sm7x/ptx_mma.cuh"

namespace tk_sm7x::detail {

template <class InstructionArch>
struct ptx_warp_mma_f16_f16_f32_16x16x16 {
    static_assert(!std::is_same<InstructionArch, InstructionArch>::value,
                  "PTX MMA backend is not supported by the active target");
};

// Every operation is warp-collective and requires all 32 lanes converged. The
// caller synchronizes shared writes before loads, completed reads before reuse,
// and reads after stores. Shared bases are 32-byte aligned; A/B ldm is at least
// 16 and a multiple of 8 halves, while C ldm is at least 16 and a multiple of 4
// floats.
template <>
struct ptx_warp_mma_f16_f16_f32_16x16x16<arch::sm70> {
    static_assert(arch::traits<arch::target>::has_mma_m8n8k4,
                  "SM70 PTX MMA backend requires m8n8k4");

    struct fragment_a {
        uint32_t value[4][2];
    };

    struct fragment_b {
        uint32_t value[4][2];
    };

    struct accumulator {
        float value[8];
    };

private:
    __device__ static __forceinline__ void load_operand(
        uint32_t (&destination)[4][2], const __half* shared) {
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
#pragma unroll
            for (int reg = 0; reg < 2; ++reg) {
                __half halves[2];
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int k =
                        4 * kq + mma_m8n8k4::operand_slot(reg, half);
                    halves[half] = shared[k];
                }
                destination[kq][reg] = pack_halves(halves[0], halves[1]);
            }
        }
    }

public:
    __device__ static __forceinline__ void load_a(
        fragment_a& destination, const __half* shared, int ldm) {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        const int mh = mma_m8n8k4::quadpair(lane) / 2;
        const int pos = mma_m8n8k4::position(lane);
        const std::size_t offset =
            static_cast<std::size_t>(8 * mh + pos) *
            static_cast<std::size_t>(ldm);
        load_operand(destination.value, shared + offset);
    }

    __device__ static __forceinline__ void load_b(
        fragment_b& destination, const __half* shared, int ldm) {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        const int nh = mma_m8n8k4::quadpair(lane) % 2;
        const int pos = mma_m8n8k4::position(lane);
        const std::size_t offset =
            static_cast<std::size_t>(8 * nh + pos) *
            static_cast<std::size_t>(ldm);
        load_operand(destination.value, shared + offset);
    }

    __device__ static __forceinline__ void clear(accumulator& destination) {
#pragma unroll
        for (int reg = 0; reg < 8; ++reg) {
            destination.value[reg] = 0.0f;
        }
    }

    __device__ static __forceinline__ void mma(
        accumulator& destination, const fragment_a& a, const fragment_b& b) {
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
            mma_m8n8k4::mma(destination.value, a.value[kq][0], a.value[kq][1],
                            b.value[kq][0], b.value[kq][1]);
        }
    }

    __device__ static __forceinline__ void store(
        float* shared, const accumulator& source, int ldm) {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        const int quadpair = mma_m8n8k4::quadpair(lane);
        const int mh = quadpair / 2;
        const int nh = quadpair % 2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int local_row = (lane & 1) + (i & 2) + 4 * (lane / 16);
            const int local_col = (i & 4) + (lane & 2) + (i & 1);
            const std::size_t offset =
                static_cast<std::size_t>(8 * mh + local_row) *
                    static_cast<std::size_t>(ldm) +
                static_cast<std::size_t>(8 * nh + local_col);
            shared[offset] = source.value[i];
        }
    }
};

#if defined(KITTENS_SM75)
template <>
struct ptx_warp_mma_f16_f16_f32_16x16x16<arch::sm75> {
    static_assert(arch::traits<arch::target>::has_mma_m16n8k8,
                  "SM75 PTX MMA backend requires m16n8k8");
    static_assert(arch::traits<arch::target>::has_ldmatrix,
                  "SM75 PTX MMA backend requires ldmatrix");

    struct fragment_a {
        uint32_t value[2][2];
    };

    struct fragment_b {
        uint32_t value[2][2];
    };

    struct accumulator {
        float value[2][4];
    };

    __device__ static __forceinline__ void load_a(
        fragment_a& destination, const __half* shared, int ldm) {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        const int owner = lane % 16;
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const std::size_t offset =
                static_cast<std::size_t>(owner) * static_cast<std::size_t>(ldm) +
                static_cast<std::size_t>(kh) * 8;
            ldmatrix_x2(destination.value[kh], shared_address(shared + offset));
        }
    }

    __device__ static __forceinline__ void load_b(
        fragment_b& destination, const __half* shared, int ldm) {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        const int owner = lane % 8;
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
#pragma unroll
            for (int nh = 0; nh < 2; ++nh) {
                const std::size_t offset =
                    static_cast<std::size_t>(owner + nh * 8) *
                        static_cast<std::size_t>(ldm) +
                    static_cast<std::size_t>(kh) * 8;
                ldmatrix_x1(destination.value[kh][nh],
                            shared_address(shared + offset));
            }
        }
    }

    __device__ static __forceinline__ void clear(accumulator& destination) {
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
#pragma unroll
            for (int reg = 0; reg < 4; ++reg) {
                destination.value[nh][reg] = 0.0f;
            }
        }
    }

    __device__ static __forceinline__ void mma(
        accumulator& destination, const fragment_a& a, const fragment_b& b) {
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
#pragma unroll
            for (int nh = 0; nh < 2; ++nh) {
                mma_m16n8k8::mma(destination.value[nh], a.value[kh][0],
                                 a.value[kh][1], b.value[kh][nh]);
            }
        }
    }

    __device__ static __forceinline__ void store(
        float* shared, const accumulator& source, int ldm) {
        const int lane = static_cast<int>(threadIdx.x) % 32;
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
#pragma unroll
            for (int reg = 0; reg < 4; ++reg) {
                const std::size_t offset =
                    static_cast<std::size_t>(
                        mma_m16n8k8::accumulator_row(lane, reg)) *
                        static_cast<std::size_t>(ldm) +
                    static_cast<std::size_t>(8 * nh +
                        mma_m16n8k8::accumulator_col(lane, reg));
                shared[offset] = source.value[nh][reg];
            }
        }
    }
};
#endif

}  // namespace tk_sm7x::detail
