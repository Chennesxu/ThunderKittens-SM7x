#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>

#include "test_utils.cuh"
#include "tk_sm7x/mma.cuh"

namespace {

__global__ void mma_correctness_kernel(
    const __half* a, const __half* b, float* c) {
    __shared__ __align__(32) __half a_shared[256];
    __shared__ __align__(32) __half b_shared[256];
    __shared__ __align__(32) float c_shared[256];

    const int lane = static_cast<int>(threadIdx.x);
    for (int linear = lane; linear < 256; linear += 32) {
        const int row = linear / 16;
        const int col = linear % 16;
        a_shared[linear] = a[linear];
        b_shared[col * 16 + row] = b[linear];
    }
    __syncwarp(0xffffffffu);

    tk_sm7x::detail::active_warp_mma::accumulator accumulator;
    tk_sm7x::detail::active_warp_mma::clear(accumulator);
    tk_sm7x::detail::active_warp_mma::mma(accumulator, a_shared, b_shared);
    tk_sm7x::detail::active_warp_mma::store(c_shared, accumulator);
    __syncwarp(0xffffffffu);

    for (int linear = lane; linear < 256; linear += 32) {
        c[linear] = c_shared[linear];
    }
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selection_status = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selection_status != EXIT_SUCCESS) {
        return selection_status;
    }

    std::array<__half, 256> a{};
    std::array<__half, 256> b{};
    std::array<float, 256> actual{};
    std::array<float, 256> reference{};
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            a[row * 16 + col] = __float2half(static_cast<float>((row * 3 + col * 5) % 7 - 3));
            b[row * 16 + col] = __float2half(static_cast<float>((row * 2 + col * 3) % 5 - 2));
        }
    }
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            double sum = 0.0;
            for (int kk = 0; kk < 16; ++kk) {
                sum += static_cast<double>(__half2float(a[row * 16 + kk])) *
                       static_cast<double>(__half2float(b[kk * 16 + col]));
            }
            reference[row * 16 + col] = static_cast<float>(sum);
        }
    }

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_c = nullptr;
    const auto release = [&]() {
        bool released = true;
        if (device_a != nullptr) {
            released = tk_sm7x::test::cuda_ok(cudaFree(device_a), "cudaFree(A)") && released;
            device_a = nullptr;
        }
        if (device_b != nullptr) {
            released = tk_sm7x::test::cuda_ok(cudaFree(device_b), "cudaFree(B)") && released;
            device_b = nullptr;
        }
        if (device_c != nullptr) {
            released = tk_sm7x::test::cuda_ok(cudaFree(device_c), "cudaFree(C)") && released;
            device_c = nullptr;
        }
        return released;
    };

    if (!tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_a), sizeof(a)),
                                "cudaMalloc(A)") ||
        !tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_b), sizeof(b)),
                                "cudaMalloc(B)") ||
        !tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_c), sizeof(actual)),
                                "cudaMalloc(C)") ||
        !tk_sm7x::test::cuda_ok(cudaMemcpy(device_a, a.data(), sizeof(a), cudaMemcpyHostToDevice),
                                "cudaMemcpy(A)") ||
        !tk_sm7x::test::cuda_ok(cudaMemcpy(device_b, b.data(), sizeof(b), cudaMemcpyHostToDevice),
                                "cudaMemcpy(B)")) {
        static_cast<void>(release());
        return EXIT_FAILURE;
    }

    static_cast<void>(cudaGetLastError());
    mma_correctness_kernel<<<1, 32>>>(device_a, device_b, device_c);
    if (!tk_sm7x::test::cuda_ok(cudaGetLastError(), "mma_correctness_kernel launch") ||
        !tk_sm7x::test::cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
        !tk_sm7x::test::cuda_ok(
            cudaMemcpy(actual.data(), device_c, sizeof(actual), cudaMemcpyDeviceToHost),
            "cudaMemcpy(C)")) {
        static_cast<void>(release());
        return EXIT_FAILURE;
    }

    for (int index = 0; index < 256; ++index) {
        if (actual[index] != reference[index]) {
            std::fprintf(stderr,
                         "mma mismatch row=%d col=%d actual=%g reference=%g\n",
                         index / 16, index % 16, actual[index], reference[index]);
            static_cast<void>(release());
            return EXIT_FAILURE;
        }
    }

    if (!release()) {
        return EXIT_FAILURE;
    }
    std::printf("mma numerical: PASS ordinal=%d\n", ordinal);
    return EXIT_SUCCESS;
}
