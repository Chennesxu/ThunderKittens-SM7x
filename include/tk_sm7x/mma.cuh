#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#include <type_traits>

#include "tk_sm7x/arch.cuh"

namespace tk_sm7x::detail {

template <class Arch>
struct warp_mma_f16_f16_f32_16x16x16 {
    static_assert(std::is_same<Arch, arch::target>::value,
                  "MMA backend architecture must match active target");

    struct accumulator {
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> value;
    };

    __device__ static __forceinline__ void clear(accumulator& acc) {
        nvcuda::wmma::fill_fragment(acc.value, 0.0f);
    }

    __device__ static __forceinline__ void mma(
        accumulator& acc, const __half* a_shared, const __half* b_shared) {
        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a;
        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b;
        nvcuda::wmma::load_matrix_sync(a, a_shared, 16);
        nvcuda::wmma::load_matrix_sync(b, b_shared, 16);
        nvcuda::wmma::mma_sync(acc.value, a, b, acc.value);
    }

    __device__ static __forceinline__ void store(
        float* c_shared, const accumulator& acc) {
        nvcuda::wmma::store_matrix_sync(
            c_shared, acc.value, 16, nvcuda::wmma::mem_row_major);
    }
};

using active_warp_mma = warp_mma_f16_f16_f32_16x16x16<arch::target>;

}  // namespace tk_sm7x::detail
