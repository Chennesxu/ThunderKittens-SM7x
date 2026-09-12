#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "tk_sm7x/arch.cuh"

namespace tk_sm7x::detail {

#if defined(KITTENS_SM75)
__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// All 32 lanes must execute converged with valid, 16-byte-aligned shared rows.
// Prior shared writes require caller synchronization; the clobber only constrains
// compiler reordering.
__device__ __forceinline__ void ldmatrix_x2(
    uint32_t (&destination)[2], uint32_t address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(destination[0]), "=r"(destination[1])
        : "r"(address)
        : "memory");
}

// All 32 lanes must execute converged with valid, 16-byte-aligned shared rows.
// Prior shared writes require caller synchronization; the clobber only constrains
// compiler reordering.
__device__ __forceinline__ void ldmatrix_x1(
    uint32_t& destination, uint32_t address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(destination)
        : "r"(address)
        : "memory");
}
#endif

}  // namespace tk_sm7x::detail
