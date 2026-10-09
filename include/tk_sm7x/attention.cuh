#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace tk_sm7x {

// Q, K, V are contiguous row-major 16x16 FP16 tiles; out is a separate
// contiguous row-major 16x16 FP32 tile, all accessible by the selected device.
// Inputs may alias each other but not out. All storage remains alive until
// stream completion; before then, inputs are not modified and out is not read
// or written except through operations ordered with this call on stream.
// The operation enqueues asynchronously; successful return covers argument
// validation and launch only. Inputs must be finite.
cudaError_t attention_forward_16x16(
    const __half* q, const __half* k, const __half* v,
    float* out, cudaStream_t stream = nullptr);

}  // namespace tk_sm7x
