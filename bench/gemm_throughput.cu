#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "test_utils.cuh"
#include "tk_sm7x/gemm.cuh"

namespace {

using tk_sm7x::test::cuda_ok;

struct shape {
    int m;
    int n;
    int k;
};

constexpr shape kShapes[] = {
    {512, 512, 512},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
};

constexpr int kWarmupIterations = 2;
constexpr int kTimedIterations = 5;

bool release(__half* a, __half* b, float* c) {
    bool released = cuda_ok(cudaFree(a), "cudaFree(a)");
    released = cuda_ok(cudaFree(b), "cudaFree(b)") && released;
    return cuda_ok(cudaFree(c), "cudaFree(c)") && released;
}

bool measure(const shape& s, double* gflops) {
    const std::size_t a_count = static_cast<std::size_t>(s.m) * s.k;
    const std::size_t b_count = static_cast<std::size_t>(s.k) * s.n;
    const std::size_t c_count = static_cast<std::size_t>(s.m) * s.n;

    std::vector<__half> host_a(a_count);
    std::vector<__half> host_b(b_count);
    for (std::size_t i = 0; i < a_count; ++i) {
        host_a[i] = __float2half(static_cast<float>(i % 7) * 0.125f);
    }
    for (std::size_t i = 0; i < b_count; ++i) {
        host_b[i] = __float2half(static_cast<float>(i % 5) * 0.25f);
    }

    __half* a = nullptr;
    __half* b = nullptr;
    float* c = nullptr;
    if (!cuda_ok(cudaMalloc(&a, a_count * sizeof(__half)), "cudaMalloc(a)") ||
        !cuda_ok(cudaMalloc(&b, b_count * sizeof(__half)), "cudaMalloc(b)") ||
        !cuda_ok(cudaMalloc(&c, c_count * sizeof(float)), "cudaMalloc(c)")) {
        static_cast<void>(release(a, b, c));
        return false;
    }

    if (!cuda_ok(cudaMemcpy(a, host_a.data(), a_count * sizeof(__half),
                            cudaMemcpyHostToDevice), "cudaMemcpy(a)") ||
        !cuda_ok(cudaMemcpy(b, host_b.data(), b_count * sizeof(__half),
                            cudaMemcpyHostToDevice), "cudaMemcpy(b)")) {
        static_cast<void>(release(a, b, c));
        return false;
    }

    for (int i = 0; i < kWarmupIterations; ++i) {
        if (!cuda_ok(tk_sm7x::gemm_f16_f16_f32_nn(s.m, s.n, s.k, a, s.k, b, s.n, c, s.n),
                     "gemm launch")) {
            static_cast<void>(release(a, b, c));
            return false;
        }
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize")) {
        static_cast<void>(release(a, b, c));
        return false;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "cudaEventCreate(start)")) {
        static_cast<void>(release(a, b, c));
        return false;
    }
    if (!cuda_ok(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        static_cast<void>(cudaEventDestroy(start));
        static_cast<void>(release(a, b, c));
        return false;
    }

    float best_ms = 0.0f;
    bool measured = true;
    for (int i = 0; i < kTimedIterations && measured; ++i) {
        measured = cuda_ok(cudaEventRecord(start), "cudaEventRecord(start)") &&
                   cuda_ok(tk_sm7x::gemm_f16_f16_f32_nn(s.m, s.n, s.k, a, s.k, b, s.n,
                                                        c, s.n), "gemm launch") &&
                   cuda_ok(cudaEventRecord(stop), "cudaEventRecord(stop)") &&
                   cuda_ok(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float elapsed_ms = 0.0f;
        measured = measured &&
                   cuda_ok(cudaEventElapsedTime(&elapsed_ms, start, stop),
                           "cudaEventElapsedTime");
        if (measured && (i == 0 || elapsed_ms < best_ms)) {
            best_ms = elapsed_ms;
        }
    }

    bool cleaned = cuda_ok(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cleaned = cuda_ok(cudaEventDestroy(stop), "cudaEventDestroy(stop)") && cleaned;
    cleaned = release(a, b, c) && cleaned;

    if (!measured || !cleaned || best_ms <= 0.0f) {
        return false;
    }
    const double flop = 2.0 * s.m * s.n * s.k;
    *gflops = flop / (static_cast<double>(best_ms) * 1.0e6);
    return true;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selection = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selection != EXIT_SUCCESS) {
        return selection;
    }

    for (const shape& s : kShapes) {
        double gflops = 0.0;
        if (!measure(s, &gflops)) {
            std::fprintf(stderr, "benchmark failed at m=%d n=%d k=%d\n", s.m, s.n, s.k);
            return EXIT_FAILURE;
        }
        std::printf("gemm m=%-5d n=%-5d k=%-5d  %8.2f GFLOP/s\n", s.m, s.n, s.k, gflops);
    }

    std::printf("gemm throughput: DONE ordinal=%d\n", ordinal);
    return EXIT_SUCCESS;
}
