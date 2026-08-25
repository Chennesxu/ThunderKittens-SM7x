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
// rows and cols bound the addressable region: staging reads outside it yield a
// zero element and stores outside it are dropped, which lets a tile shape that
// does not divide the operand still be processed by one uniform path.
template <class T>
struct gl {
    T* data;
    int rows;
    int cols;
    int ld;
};

// Shared-memory tile. Alignment satisfies the 256-bit requirement of the
// architecture MMA fragment loads.
template <class T, int R, int C, class Layout = row_major>
struct alignas(32) st {
    static_assert(std::is_same<Layout, row_major>::value ||
                      std::is_same<Layout, col_major>::value,
                  "shared tile layout must be row_major or col_major");

    T data[static_cast<std::size_t>(R) * C];

    // linear enumerates the tile in row-major traversal order, which is the
    // order that keeps consecutive lanes on consecutive global addresses.
    __device__ static __forceinline__ constexpr int offset(int linear) {
        return std::is_same<Layout, row_major>::value
                   ? linear
                   : (linear % C) * R + linear / C;
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

// Staging between global and shared memory is CTA-collective: every thread of
// the block must reach it, and the caller must __syncthreads() before the staged
// tile is read. The register-tile operations below are warp-collective instead.
// THREADS must equal the launch block size so the per-thread trip count stays a
// compile-time constant.

template <int THREADS, class T, int R, int C, class Layout>
__device__ __forceinline__ void load_block(
    st<T, R, C, Layout>& dst, const gl<const T>& src,
    std::size_t row0, std::size_t col0) {
    static_assert(R * C % THREADS == 0, "THREADS must divide the tile element count");
#pragma unroll
    for (int linear = static_cast<int>(threadIdx.x); linear < R * C; linear += THREADS) {
        const std::size_t row = row0 + static_cast<std::size_t>(linear / C);
        const std::size_t col = col0 + static_cast<std::size_t>(linear % C);
        const bool inside = row < static_cast<std::size_t>(src.rows) &&
                            col < static_cast<std::size_t>(src.cols);
        dst.data[st<T, R, C, Layout>::offset(linear)] =
            inside ? src.data[row * static_cast<std::size_t>(src.ld) + col] : T{};
    }
}

template <int THREADS, class T, int R, int C, class Layout>
__device__ __forceinline__ void store_block(
    const gl<T>& dst, const st<T, R, C, Layout>& src,
    std::size_t row0, std::size_t col0) {
    static_assert(R * C % THREADS == 0, "THREADS must divide the tile element count");
#pragma unroll
    for (int linear = static_cast<int>(threadIdx.x); linear < R * C; linear += THREADS) {
        const std::size_t row = row0 + static_cast<std::size_t>(linear / C);
        const std::size_t col = col0 + static_cast<std::size_t>(linear % C);
        if (row < static_cast<std::size_t>(dst.rows) &&
            col < static_cast<std::size_t>(dst.cols)) {
            dst.data[row * static_cast<std::size_t>(dst.ld) + col] =
                src.data[st<T, R, C, Layout>::offset(linear)];
        }
    }
}

// Warp-collective counterpart of store_block, for a tile owned by one warp
// rather than by the whole block.
template <class T, int R, int C, class Layout>
__device__ __forceinline__ void store_warp(
    const gl<T>& dst, const st<T, R, C, Layout>& src,
    std::size_t row0, std::size_t col0) {
    static_assert(R * C % 32 == 0, "the warp width must divide the tile element count");
#pragma unroll
    for (int linear = static_cast<int>(threadIdx.x % 32u); linear < R * C; linear += 32) {
        const std::size_t row = row0 + static_cast<std::size_t>(linear / C);
        const std::size_t col = col0 + static_cast<std::size_t>(linear % C);
        if (row < static_cast<std::size_t>(dst.rows) &&
            col < static_cast<std::size_t>(dst.cols)) {
            dst.data[row * static_cast<std::size_t>(dst.ld) + col] =
                src.data[st<T, R, C, Layout>::offset(linear)];
        }
    }
}

// A 16x16 sub-view of a staged tile. Row blocks of a row-major R-by-16 tile and
// column blocks of a column-major 16-by-C tile are both contiguous runs of 256
// elements with a leading dimension of 16.
template <int R>
__device__ __forceinline__ void load(
    rt_a& dst, const st<__half, R, 16, row_major>& src, int row_block) {
    detail::active_warp_mma::load_a(dst.value, src.data + row_block * 256, 16);
}

template <int C>
__device__ __forceinline__ void load(
    rt_b& dst, const st<__half, 16, C, col_major>& src, int col_block) {
    detail::active_warp_mma::load_b(dst.value, src.data + col_block * 256, 16);
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
