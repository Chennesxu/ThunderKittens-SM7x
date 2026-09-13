#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#include <type_traits>

#include "tk_sm7x/arch.cuh"
#if defined(KITTENS_MMA_PTX)
#include "tk_sm7x/ptx_backend.cuh"
#endif

namespace tk_sm7x::detail {

template <class Arch>
struct warp_mma_f16_f16_f32_16x16x16 {
    static_assert(std::is_same<Arch, arch::target>::value,
                  "MMA backend architecture must match active target");

    struct fragment_a {
        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> value;
    };

    struct fragment_b {
        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> value;
    };

    struct accumulator {
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> value;
    };

    // Every operation is warp-collective: all 32 lanes must reach it converged.
    // Shared-memory operands must be 256-bit aligned.
    __device__ static __forceinline__ void load_a(
        fragment_a& fragment, const __half* shared, int ldm) {
        nvcuda::wmma::load_matrix_sync(fragment.value, shared, ldm);
    }

    __device__ static __forceinline__ void load_b(
        fragment_b& fragment, const __half* shared, int ldm) {
        nvcuda::wmma::load_matrix_sync(fragment.value, shared, ldm);
    }

    __device__ static __forceinline__ void clear(accumulator& acc) {
        nvcuda::wmma::fill_fragment(acc.value, 0.0f);
    }

    __device__ static __forceinline__ void mma(
        accumulator& acc, const fragment_a& a, const fragment_b& b) {
        nvcuda::wmma::mma_sync(acc.value, a.value, b.value, acc.value);
    }

    __device__ static __forceinline__ void store(
        float* shared, const accumulator& acc, int ldm) {
        nvcuda::wmma::store_matrix_sync(shared, acc.value, ldm, nvcuda::wmma::mem_row_major);
    }
};

#if defined(KITTENS_MMA_PTX)
using active_warp_mma = ptx_warp_mma_f16_f16_f32_16x16x16<arch::target>;
#else
using active_warp_mma = warp_mma_f16_f16_f32_16x16x16<arch::target>;
#endif

}  // namespace tk_sm7x::detail
