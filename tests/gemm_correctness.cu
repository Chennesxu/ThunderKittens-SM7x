#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "test_utils.cuh"
#include "tk_sm7x/gemm.cuh"

namespace {

constexpr float kSentinel = -123456.0f;

enum class Pattern {
    identity_right,
    fingerprint,
    random,
};

struct GemmCase {
    const char* name;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    Pattern pattern;
    unsigned int seed;
    bool exact;
    bool seed_stale_error;
};

struct GemmArguments {
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

__global__ void seed_stale_launch_error() {}

void fill_inputs(const GemmCase& test_case,
                 std::vector<__half>* a,
                 std::vector<__half>* b) {
    const __half zero = __float2half(0.0f);
    std::fill(a->begin(), a->end(), zero);
    std::fill(b->begin(), b->end(), zero);

    if (test_case.pattern == Pattern::random) {
        std::mt19937 generator(test_case.seed);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (int row = 0; row < test_case.m; ++row) {
            for (int col = 0; col < test_case.k; ++col) {
                (*a)[row * test_case.lda + col] = __float2half(distribution(generator));
            }
        }
        for (int row = 0; row < test_case.k; ++row) {
            for (int col = 0; col < test_case.n; ++col) {
                (*b)[row * test_case.ldb + col] = __float2half(distribution(generator));
            }
        }
        return;
    }

    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.k; ++col) {
            (*a)[row * test_case.lda + col] = __float2half(
                static_cast<float>((row * 3 + col * 5) % 7 - 3));
        }
    }
    for (int row = 0; row < test_case.k; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const float value = test_case.pattern == Pattern::identity_right
                                    ? (row == col ? 1.0f : 0.0f)
                                    : static_cast<float>((row * 2 + col * 3) % 5 - 2);
            (*b)[row * test_case.ldb + col] = __float2half(value);
        }
    }
}

std::vector<float> make_reference(const GemmCase& test_case,
                                  const std::vector<__half>& a,
                                  const std::vector<__half>& b) {
    std::vector<float> reference(
        static_cast<std::size_t>(test_case.m) * test_case.ldc, kSentinel);
    if (test_case.pattern == Pattern::identity_right) {
        for (int row = 0; row < test_case.m; ++row) {
            for (int col = 0; col < test_case.n; ++col) {
                reference[static_cast<std::size_t>(row) * test_case.ldc + col] =
                    __half2float(a[static_cast<std::size_t>(row) * test_case.lda + col]);
            }
        }
        return reference;
    }
    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            double sum = 0.0;
            for (int kk = 0; kk < test_case.k; ++kk) {
                sum += static_cast<double>(__half2float(a[row * test_case.lda + kk])) *
                       static_cast<double>(__half2float(b[kk * test_case.ldb + col]));
            }
            reference[row * test_case.ldc + col] = static_cast<float>(sum);
        }
    }
    return reference;
}

bool compare_output(const GemmCase& test_case,
                    const std::vector<float>& actual,
                    const std::vector<float>& reference) {
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    bool matched = true;
    bool first_reported = false;

    for (int row = 0; row < test_case.m; ++row) {
        for (int col = 0; col < test_case.n; ++col) {
            const int index = row * test_case.ldc + col;
            const float absolute_error = std::fabs(actual[index] - reference[index]);
            const float relative_error = absolute_error /
                std::max(std::fabs(reference[index]), 1.0e-12f);
            max_absolute_error = std::max(max_absolute_error, absolute_error);
            max_relative_error = std::max(max_relative_error, relative_error);
            const bool element_matches = test_case.exact
                ? actual[index] == reference[index]
                : absolute_error <= 2.0e-3f + 2.0e-3f * std::fabs(reference[index]);
            if (!element_matches) {
                matched = false;
                if (!first_reported) {
                    std::fprintf(stderr,
                                 "%s first mismatch row=%d col=%d actual=%g reference=%g\n",
                                 test_case.name, row, col, actual[index], reference[index]);
                    first_reported = true;
                }
            }
        }
        for (int col = test_case.n; col < test_case.ldc; ++col) {
            const int index = row * test_case.ldc + col;
            if (actual[index] != kSentinel) {
                std::fprintf(stderr,
                             "%s modified C padding row=%d col=%d actual=%g\n",
                             test_case.name, row, col, actual[index]);
                matched = false;
            }
        }
    }

    if (!matched) {
        std::fprintf(stderr, "%s max_abs=%g max_rel=%g\n", test_case.name,
                     max_absolute_error, max_relative_error);
    }
    return matched;
}

bool run_case(const GemmCase& test_case, cudaStream_t stream) {
    std::vector<__half> a(
        static_cast<std::size_t>(test_case.m) * test_case.lda);
    std::vector<__half> b(
        static_cast<std::size_t>(test_case.k) * test_case.ldb);
    std::vector<float> actual(
        static_cast<std::size_t>(test_case.m) * test_case.ldc, kSentinel);
    fill_inputs(test_case, &a, &b);
    const std::vector<float> reference = make_reference(test_case, a, b);

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

    const std::size_t a_bytes = a.size() * sizeof(__half);
    const std::size_t b_bytes = b.size() * sizeof(__half);
    const std::size_t c_bytes = actual.size() * sizeof(float);
    if (!tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_a), a_bytes),
                                "cudaMalloc(A)") ||
        !tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_b), b_bytes),
                                "cudaMalloc(B)") ||
        !tk_sm7x::test::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_c), c_bytes),
                                "cudaMalloc(C)") ||
        !tk_sm7x::test::cuda_ok(
            cudaMemcpy(device_a, a.data(), a_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(A)") ||
        !tk_sm7x::test::cuda_ok(
            cudaMemcpy(device_b, b.data(), b_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(B)") ||
        !tk_sm7x::test::cuda_ok(
            cudaMemcpy(device_c, actual.data(), c_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(C sentinel)")) {
        static_cast<void>(release());
        return false;
    }

    if (test_case.seed_stale_error) {
        seed_stale_launch_error<<<0, 1>>>();
        if (cudaPeekAtLastError() == cudaSuccess) {
            std::fprintf(stderr, "%s failed to seed stale CUDA error\n", test_case.name);
            static_cast<void>(release());
            return false;
        }
    }

    const cudaError_t launch_status = tk_sm7x::gemm_f16_f16_f32_nn(
        test_case.m, test_case.n, test_case.k,
        device_a, test_case.lda, device_b, test_case.ldb,
        device_c, test_case.ldc, stream);
    if (!tk_sm7x::test::cuda_ok(launch_status, test_case.name) ||
        !tk_sm7x::test::cuda_ok(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize") ||
        !tk_sm7x::test::cuda_ok(
            cudaMemcpy(actual.data(), device_c, c_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(C)")) {
        static_cast<void>(release());
        return false;
    }

    const bool matched = compare_output(test_case, actual, reference);
    if (!matched) {
        static_cast<void>(release());
        return false;
    }
    if (!release()) {
        return false;
    }
    std::printf("%s: PASS\n", test_case.name);
    return true;
}

bool expect_invalid(const char* name,
                    const GemmArguments& arguments,
                    float* observed_c,
                    std::size_t c_elements,
                    cudaStream_t stream,
                    bool seed_stale_error) {
    std::vector<float> sentinel(c_elements, kSentinel);
    if (!tk_sm7x::test::cuda_ok(
            cudaMemcpy(observed_c, sentinel.data(), c_elements * sizeof(float),
                       cudaMemcpyHostToDevice),
            "reset invalid-case C")) {
        return false;
    }

    static_cast<void>(cudaGetLastError());
    cudaError_t stale_status = cudaSuccess;
    if (seed_stale_error) {
        seed_stale_launch_error<<<0, 1>>>();
        stale_status = cudaPeekAtLastError();
        if (stale_status == cudaSuccess) {
            std::fprintf(stderr, "%s failed to seed stale CUDA error\n", name);
            return false;
        }
    }
    const cudaError_t status = tk_sm7x::gemm_f16_f16_f32_nn(
        arguments.m, arguments.n, arguments.k,
        arguments.a, arguments.lda, arguments.b, arguments.ldb,
        arguments.c, arguments.ldc, stream);
    bool passed = true;
    if (status != cudaErrorInvalidValue) {
        std::fprintf(stderr, "%s returned %s instead of cudaErrorInvalidValue\n",
                     name, cudaGetErrorString(status));
        passed = false;
    }
    if (seed_stale_error) {
        const cudaError_t preserved_status = cudaPeekAtLastError();
        if (preserved_status != stale_status) {
            std::fprintf(stderr, "%s did not preserve stale CUDA error\n", name);
            passed = false;
        }
        if (cudaGetLastError() != stale_status ||
            cudaPeekAtLastError() != cudaSuccess) {
            std::fprintf(stderr, "%s stale CUDA error was not consumed exactly once\n", name);
            passed = false;
        }
    }
    if (!tk_sm7x::test::cuda_ok(cudaStreamSynchronize(stream),
                                "invalid-case cudaStreamSynchronize") ||
        !tk_sm7x::test::cuda_ok(
            cudaMemcpy(sentinel.data(), observed_c, c_elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "read invalid-case C")) {
        return false;
    }
    for (float value : sentinel) {
        if (value != kSentinel) {
            std::fprintf(stderr, "%s modified output despite invalid arguments\n", name);
            passed = false;
            break;
        }
    }
    return passed;
}

bool run_invalid_cases(cudaStream_t stream) {
    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_c = nullptr;
    constexpr std::size_t elements = 256;
    const auto release = [&]() {
        bool released = true;
        if (device_a != nullptr) {
            released = tk_sm7x::test::cuda_ok(
                cudaFree(device_a), "cudaFree(invalid A)") && released;
            device_a = nullptr;
        }
        if (device_b != nullptr) {
            released = tk_sm7x::test::cuda_ok(
                cudaFree(device_b), "cudaFree(invalid B)") && released;
            device_b = nullptr;
        }
        if (device_c != nullptr) {
            released = tk_sm7x::test::cuda_ok(
                cudaFree(device_c), "cudaFree(invalid C)") && released;
            device_c = nullptr;
        }
        return released;
    };
    if (!tk_sm7x::test::cuda_ok(
            cudaMalloc(reinterpret_cast<void**>(&device_a), elements * sizeof(__half)),
            "cudaMalloc(invalid A)") ||
        !tk_sm7x::test::cuda_ok(
            cudaMalloc(reinterpret_cast<void**>(&device_b), elements * sizeof(__half)),
            "cudaMalloc(invalid B)") ||
        !tk_sm7x::test::cuda_ok(
            cudaMalloc(reinterpret_cast<void**>(&device_c), elements * sizeof(float)),
            "cudaMalloc(invalid C)")) {
        static_cast<void>(release());
        return false;
    }

    const GemmArguments base{16, 16, 16, device_a, 16, device_b, 16, device_c, 16};
    bool passed = true;
    const auto check = [&](const char* name, const GemmArguments& arguments) {
        passed = expect_invalid(name, arguments, device_c, elements, stream, false) && passed;
    };
    const auto check_preserving_stale_error =
        [&](const char* name, const GemmArguments& arguments) {
            passed = expect_invalid(name, arguments, device_c, elements, stream, true) && passed;
    };

    GemmArguments arguments = base;
    arguments.a = nullptr;
    check_preserving_stale_error("null A with stale error", arguments);
    arguments = base;
    arguments.b = nullptr;
    check("null B", arguments);
    arguments = base;
    arguments.c = nullptr;
    check("null C", arguments);
    arguments = base;
    arguments.m = 0;
    check("zero M", arguments);
    arguments = base;
    arguments.n = 0;
    check("zero N", arguments);
    arguments = base;
    arguments.k = 0;
    check("zero K", arguments);
    arguments = base;
    arguments.m = -16;
    check("negative M", arguments);
    arguments = base;
    arguments.n = -16;
    check("negative N", arguments);
    arguments = base;
    arguments.k = -16;
    check("negative K", arguments);
    arguments = base;
    arguments.m = 17;
    check("nonmultiple M", arguments);
    arguments = base;
    arguments.n = 17;
    check("nonmultiple N", arguments);
    arguments = base;
    arguments.k = 17;
    check("nonmultiple K", arguments);
    arguments = base;
    arguments.lda = 15;
    check("short lda", arguments);
    arguments = base;
    arguments.ldb = 15;
    check("short ldb", arguments);
    arguments = base;
    arguments.ldc = 15;
    check("short ldc", arguments);

    if (!passed) {
        static_cast<void>(release());
        return false;
    }
    if (!release()) {
        return false;
    }
    std::printf("invalid argument matrix: PASS\n");
    return true;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selection_status = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selection_status != EXIT_SUCCESS) {
        return selection_status;
    }

    cudaStream_t stream = nullptr;
    if (!tk_sm7x::test::cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        return EXIT_FAILURE;
    }

    const GemmCase cases[] = {
        {"identity-16x16x16", 16, 16, 16, 16, 16, 16,
         Pattern::identity_right, 0u, true, true},
        {"identity-8388608x16x16-grid-y-boundary", 8388608, 16, 16, 16, 16, 16,
         Pattern::identity_right, 0u, true, false},
        {"fingerprint-16x16x32", 16, 16, 32, 32, 16, 16,
         Pattern::fingerprint, 0u, true, false},
        {"random-32x48x32-strided", 32, 48, 32, 37, 53, 59,
         Pattern::random, 0x75c0ffeeu, false, false},
        {"random-32x32x48", 32, 32, 48, 48, 32, 32,
         Pattern::random, 0x70c0ffeeu, false, false},
        {"random-256x384x32-strided-multi-cta", 256, 384, 32, 41, 397, 401,
         Pattern::random, 0x5eed1234u, false, false},
    };

    bool passed = true;
    for (const GemmCase& test_case : cases) {
        passed = run_case(test_case, stream) && passed;
    }
    passed = run_invalid_cases(stream) && passed;

    if (!passed) {
        static_cast<void>(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }
    if (!tk_sm7x::test::cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy")) {
        return EXIT_FAILURE;
    }
    std::printf("GEMM contract: PASS ordinal=%d\n", ordinal);
    return EXIT_SUCCESS;
}
