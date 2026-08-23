#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace tk_sm7x {

// A, B, and C are non-overlapping row-major allocations accessible from the
// selected CUDA device and must remain alive until stream completes. Before
// completion, A/B cannot be modified and C cannot be read or written except
// by accesses ordered with this GEMM on stream. This asynchronous baseline
// accepts only m=n=k=lda=ldb=ldc=16. A successful return reports argument and
// launch status; execution errors surface through later synchronization.
cudaError_t gemm_f16_f16_f32_nn(
    int m, int n, int k,
    const __half* a, int lda,
    const __half* b, int ldb,
    float* c, int ldc,
    cudaStream_t stream = nullptr);

}  // namespace tk_sm7x
