#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace tk_sm7x {

// A is m-by-k, B is k-by-n, and C is m-by-n; all are non-overlapping
// row-major allocations accessible from the selected CUDA device. Dimensions
// are positive multiples of 16 and lda>=k, ldb>=n, ldc>=n. A/B/C must remain
// alive until stream completes. Before completion, A/B cannot be modified and
// C cannot be read or written except by accesses ordered with this GEMM on
// stream. A successful asynchronous return covers argument validation and
// this launch; execution errors surface through later synchronization APIs.
cudaError_t gemm_f16_f16_f32_nn(
    int m, int n, int k,
    const __half* a, int lda,
    const __half* b, int ldb,
    float* c, int ldc,
    cudaStream_t stream = nullptr);

}  // namespace tk_sm7x
