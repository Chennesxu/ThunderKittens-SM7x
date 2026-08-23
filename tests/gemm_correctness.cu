#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>

#include "test_utils.cuh"
#include "tk_sm7x/gemm.cuh"

namespace {

__global__ void seed_stale_launch_error() {}

bool verify_identity(const std::array<__half, 256>& a,
                     const std::array<float, 256>& c) {
    for (int index = 0; index < 256; ++index) {
        const float reference = __half2float(a[index]);
        if (c[index] != reference) {
            std::fprintf(stderr,
                         "identity mismatch row=%d col=%d actual=%g reference=%g\n",
                         index / 16, index % 16, c[index], reference);
            return false;
        }
    }
    return true;
}

struct CompactArguments {
    int m;
    int n;
    int k;
    const __half* a;
    int lda;
    const __half* b;
    int ldb;
    float* c;
    int ldc;
};

bool expect_compact_invalid(const char* name,
                            const CompactArguments& arguments,
                            float* observed_c,
                            cudaStream_t stream) {
    std::array<float, 256> sentinel{};
    sentinel.fill(-654321.0f);
    if (!tk_sm7x::test::cuda_ok(
            cudaMemcpy(observed_c, sentinel.data(), sizeof(sentinel),
                       cudaMemcpyHostToDevice),
            "reset compact invalid C")) {
        return false;
    }
    static_cast<void>(cudaGetLastError());
    const cudaError_t status = tk_sm7x::gemm_f16_f16_f32_nn(
        arguments.m, arguments.n, arguments.k,
        arguments.a, arguments.lda, arguments.b, arguments.ldb,
        arguments.c, arguments.ldc, stream);
    bool passed = status == cudaErrorInvalidValue;
    if (!passed) {
        std::fprintf(stderr, "%s returned %s instead of cudaErrorInvalidValue\n",
                     name, cudaGetErrorString(status));
    }
    if (!tk_sm7x::test::cuda_ok(cudaStreamSynchronize(stream),
                                "compact invalid cudaStreamSynchronize") ||
        !tk_sm7x::test::cuda_ok(
            cudaMemcpy(sentinel.data(), observed_c, sizeof(sentinel),
                       cudaMemcpyDeviceToHost),
            "read compact invalid C")) {
        return false;
    }
    for (float value : sentinel) {
        if (value != -654321.0f) {
            std::fprintf(stderr, "%s modified output\n", name);
            passed = false;
            break;
        }
    }
    return passed;
}

bool run_compact_invalid_cases(const __half* device_a,
                               const __half* device_b,
                               float* device_c,
                               cudaStream_t stream) {
    const CompactArguments base{
        16, 16, 16, device_a, 16, device_b, 16, device_c, 16};
    bool passed = true;
    const auto check = [&](const char* name, const CompactArguments& arguments) {
        passed = expect_compact_invalid(name, arguments, device_c, stream) && passed;
    };
    CompactArguments arguments = base;

    arguments.a = nullptr;
    check("compact null A", arguments);
    arguments = base;
    arguments.b = nullptr;
    check("compact null B", arguments);
    arguments = base;
    arguments.c = nullptr;
    check("compact null C", arguments);

    arguments = base;
    arguments.m = 0;
    check("compact zero M", arguments);
    arguments = base;
    arguments.n = 0;
    check("compact zero N", arguments);
    arguments = base;
    arguments.k = 0;
    check("compact zero K", arguments);
    arguments = base;
    arguments.m = -16;
    check("compact negative M", arguments);
    arguments = base;
    arguments.n = -16;
    check("compact negative N", arguments);
    arguments = base;
    arguments.k = -16;
    check("compact negative K", arguments);

    arguments = base;
    arguments.m = 32;
    check("compact non-16 M", arguments);
    arguments = base;
    arguments.n = 32;
    check("compact non-16 N", arguments);
    arguments = base;
    arguments.k = 32;
    check("compact non-16 K", arguments);
    arguments = base;
    arguments.lda = 15;
    check("compact short lda", arguments);
    arguments = base;
    arguments.lda = 17;
    check("compact long lda", arguments);
    arguments = base;
    arguments.ldb = 15;
    check("compact short ldb", arguments);
    arguments = base;
    arguments.ldb = 17;
    check("compact long ldb", arguments);
    arguments = base;
    arguments.ldc = 15;
    check("compact short ldc", arguments);
    arguments = base;
    arguments.ldc = 17;
    check("compact long ldc", arguments);

    return passed;
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
    std::array<float, 256> c{};
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            a[row * 16 + col] =
                __float2half(static_cast<float>((row * 5 + col * 3) % 17 - 8));
            b[row * 16 + col] = __float2half(row == col ? 1.0f : 0.0f);
        }
    }

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_c = nullptr;
    cudaStream_t stream = nullptr;
    const auto release = [&]() {
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (device_a != nullptr) cudaFree(device_a);
        if (device_b != nullptr) cudaFree(device_b);
        if (device_c != nullptr) cudaFree(device_c);
    };

    if (!tk_sm7x::test::cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate") ||
        !tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_a), sizeof(a)),
                                "cudaMalloc(A)") ||
        !tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_b), sizeof(b)),
                                "cudaMalloc(B)") ||
        !tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_c), sizeof(c)),
                                "cudaMalloc(C)") ||
        !tk_sm7x::test::cuda_ok(cudaMemcpy(device_a, a.data(), sizeof(a), cudaMemcpyHostToDevice),
                                "cudaMemcpy(A)") ||
        !tk_sm7x::test::cuda_ok(cudaMemcpy(device_b, b.data(), sizeof(b), cudaMemcpyHostToDevice),
                                "cudaMemcpy(B)")) {
        release();
        return EXIT_FAILURE;
    }

    const auto run_and_check = [&]() {
        const cudaError_t launch_status = tk_sm7x::gemm_f16_f16_f32_nn(
            16, 16, 16, device_a, 16, device_b, 16, device_c, 16, stream);
        if (!tk_sm7x::test::cuda_ok(launch_status, "gemm_f16_f16_f32_nn") ||
            !tk_sm7x::test::cuda_ok(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize") ||
            !tk_sm7x::test::cuda_ok(
                cudaMemcpy(c.data(), device_c, sizeof(c), cudaMemcpyDeviceToHost),
                "cudaMemcpy(C)")) {
            return false;
        }
        return verify_identity(a, c);
    };

    if (!run_and_check()) {
        release();
        return EXIT_FAILURE;
    }

    seed_stale_launch_error<<<0, 1>>>();
    if (cudaPeekAtLastError() == cudaSuccess) {
        std::fprintf(stderr, "failed to seed stale CUDA last-error state\n");
        release();
        return EXIT_FAILURE;
    }
    if (!run_and_check()) {
        std::fprintf(stderr, "valid GEMM returned a stale launch error\n");
        release();
        return EXIT_FAILURE;
    }
    if (!run_compact_invalid_cases(device_a, device_b, device_c, stream)) {
        release();
        return EXIT_FAILURE;
    }

    release();
    std::printf("compact GEMM: PASS ordinal=%d\n", ordinal);
    return EXIT_SUCCESS;
}
