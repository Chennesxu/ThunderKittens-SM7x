#include "tk_sm7x/attention.cuh"

#include <cstddef>

#include "tk_sm7x/softmax.cuh"

namespace tk_sm7x {
namespace {

template <bool Causal>
__global__ void attention_kernel(const __half* q, const __half* k,
                                 const __half* v, float* out) {
    __shared__ st<__half, 16, 16, row_major> shared_q;
    __shared__ st<__half, 16, 16, col_major> shared_k_transposed;
    __shared__ st<__half, 16, 16, col_major> shared_v;
    __shared__ st<float, 16, 16, row_major> probabilities;
    __shared__ st<float, 16, 16, row_major> scratch;
    __shared__ st<__half, 16, 16, row_major> rounded_probabilities;

    load_block<32>(shared_q, gl<const __half>{q, 16, 16, 16}, 0, 0);
    load_block<32>(shared_v, gl<const __half>{v, 16, 16, 16}, 0, 0);
#pragma unroll
    for (int linear = static_cast<int>(threadIdx.x); linear < 256; linear += 32) {
        // Physical row-major K bytes are column-major K^T bytes.
        shared_k_transposed.data[linear] = k[static_cast<std::size_t>(linear)];
    }
    __syncthreads();

    rt_a q_tile;
    rt_b k_tile;
    rt_c scores;
    load(q_tile, shared_q, 0);
    load(k_tile, shared_k_transposed, 0);
    zero(scores);
    mma(scores, q_tile, k_tile);
    scale(scores, 0.25f);
    if constexpr (Causal) {
        row_softmax_causal(probabilities, scores, scratch);
    } else {
        row_softmax(probabilities, scores, scratch);
    }

#pragma unroll
    for (int linear = static_cast<int>(threadIdx.x); linear < 256; linear += 32) {
        rounded_probabilities.data[linear] =
            __float2half_rn(probabilities.data[linear]);
    }
    __syncwarp(0xffffffffu);

    rt_a probability_tile;
    rt_b v_tile;
    rt_c output_tile;
    load(probability_tile, rounded_probabilities, 0);
    load(v_tile, shared_v, 0);
    zero(output_tile);
    mma(output_tile, probability_tile, v_tile);
    store(scratch, output_tile);
    __syncwarp(0xffffffffu);
    store_block<32>(gl<float>{out, 16, 16, 16}, scratch, 0, 0);
}

template <bool Causal>
cudaError_t launch_attention(
    const __half* q, const __half* k, const __half* v,
    float* out, cudaStream_t stream) {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
        return cudaErrorInvalidValue;
    }
    static_cast<void>(cudaGetLastError());
    attention_kernel<Causal><<<1, 32, 0, stream>>>(q, k, v, out);
    return cudaGetLastError();
}

}  // namespace

cudaError_t attention_forward_16x16(
    const __half* q, const __half* k, const __half* v,
    float* out, cudaStream_t stream) {
    return launch_attention<false>(q, k, v, out, stream);
}

cudaError_t attention_forward_16x16_causal(
    const __half* q, const __half* k, const __half* v,
    float* out, cudaStream_t stream) {
    return launch_attention<true>(q, k, v, out, stream);
}

}  // namespace tk_sm7x
