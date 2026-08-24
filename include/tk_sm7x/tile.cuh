#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <type_traits>

#include "tk_sm7x/mma.cuh"

namespace tk_sm7x {

struct row_major {};
struct col_major {};

// Row-major view of device-accessible storage. Offsets are formed in
// std::size_t so the full positive dimension domain stays representable.
template <class T>
struct gl {
    T* data;
    int ld;
};

// Shared-memory tile. Alignment satisfies the 256-bit requirement of the
// architecture MMA fragment loads.
template <class T, int R, int C, class Layout = row_major>
struct alignas(32) st {
    T data[static_cast<std::size_t>(R) * C];

    // linear enumerates the tile in row-major traversal order, which is the
    // order that keeps consecutive lanes on consecutive global addresses.
    __device__ static __forceinline__ int offset(int linear) {
        if constexpr (std::is_same<Layout, row_major>::value) {
            return linear;
        } else {
            return (linear % C) * R + linear / C;
        }
    }

    __device__ static __forceinline__ constexpr int leading_dimension() {
        return std::is_same<Layout, row_major>::value ? C : R;
    }
};

// Register tiles owning one architecture MMA fragment each. Holding them as
// named values is what lets a caller reuse an operand across iterations.
struct rt_a {
    detail::active_warp_mma::fragment_a value;
};

struct rt_b {
    detail::active_warp_mma::fragment_b value;
};

struct rt_c {
    detail::active_warp_mma::accumulator value;
};

// All operations below are warp-collective: every lane of the warp must reach
// them converged.

template <class T, int R, int C, class Layout>
__device__ __forceinline__ void load(
    st<T, R, C, Layout>& dst, const gl<const T>& src,
    std::size_t row0, std::size_t col0) {
    const int lane = static_cast<int>(threadIdx.x % 32u);
    // The per-lane trip count is a compile-time constant; full unrolling is what
    // keeps the global accesses of one tile batched together.
#pragma unroll
    for (int linear = lane; linear < R * C; linear += 32) {
        const int r = linear / C;
        const int c = linear % C;
        dst.data[st<T, R, C, Layout>::offset(linear)] =
            src.data[(row0 + static_cast<std::size_t>(r)) *
                         static_cast<std::size_t>(src.ld) +
                     col0 + static_cast<std::size_t>(c)];
    }
}

template <class T, int R, int C, class Layout>
__device__ __forceinline__ void store(
    const gl<T>& dst, const st<T, R, C, Layout>& src,
    std::size_t row0, std::size_t col0) {
    const int lane = static_cast<int>(threadIdx.x % 32u);
    // The per-lane trip count is a compile-time constant; full unrolling is what
    // keeps the global accesses of one tile batched together.
#pragma unroll
    for (int linear = lane; linear < R * C; linear += 32) {
        const int r = linear / C;
        const int c = linear % C;
        dst.data[(row0 + static_cast<std::size_t>(r)) *
                     static_cast<std::size_t>(dst.ld) +
                 col0 + static_cast<std::size_t>(c)] =
            src.data[st<T, R, C, Layout>::offset(linear)];
    }
}

__device__ __forceinline__ void load(rt_a& dst, const st<__half, 16, 16, row_major>& src) {
    detail::active_warp_mma::load_a(
        dst.value, src.data, st<__half, 16, 16, row_major>::leading_dimension());
}

__device__ __forceinline__ void load(rt_b& dst, const st<__half, 16, 16, col_major>& src) {
    detail::active_warp_mma::load_b(
        dst.value, src.data, st<__half, 16, 16, col_major>::leading_dimension());
}

__device__ __forceinline__ void zero(rt_c& dst) {
    detail::active_warp_mma::clear(dst.value);
}

__device__ __forceinline__ void mma(rt_c& dst, const rt_a& a, const rt_b& b) {
    detail::active_warp_mma::mma(dst.value, a.value, b.value);
}

__device__ __forceinline__ void store(st<float, 16, 16, row_major>& dst, const rt_c& src) {
    detail::active_warp_mma::store(
        dst.data, src.value, st<float, 16, 16, row_major>::leading_dimension());
}

}  // namespace tk_sm7x
