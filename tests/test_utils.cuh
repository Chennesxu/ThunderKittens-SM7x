#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace tk_sm7x::test {

constexpr int kSkip = 77;

inline bool cuda_ok(cudaError_t status, const char* expression) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", expression, cudaGetErrorString(status));
    return false;
}

inline int select_sm75_device(int* selected_ordinal) {
    *selected_ordinal = -1;
    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess) {
        const bool unavailable = count_status == cudaErrorNoDevice ||
            count_status == cudaErrorInsufficientDriver ||
            count_status == cudaErrorSystemDriverMismatch;
        std::fprintf(stderr, "%s: cudaGetDeviceCount failed: %s\n",
                     unavailable ? "SKIP" : "ERROR", cudaGetErrorString(count_status));
        static_cast<void>(cudaGetLastError());
        return unavailable ? kSkip : EXIT_FAILURE;
    }

    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cudaDeviceProp properties{};
        const cudaError_t property_status = cudaGetDeviceProperties(&properties, ordinal);
        if (property_status != cudaSuccess) {
            std::fprintf(stderr, "cudaGetDeviceProperties(%d) failed: %s\n", ordinal,
                         cudaGetErrorString(property_status));
            return EXIT_FAILURE;
        }
        std::fprintf(stderr, "CUDA device ordinal=%d name=%s cc=%d.%d\n", ordinal,
                     properties.name, properties.major, properties.minor);
        if (*selected_ordinal < 0 && properties.major == 7 && properties.minor == 5) {
            *selected_ordinal = ordinal;
        }
    }

    if (*selected_ordinal < 0) {
        std::fprintf(stderr, "SKIP: no CUDA device with compute capability 7.5\n");
        return kSkip;
    }

    if (!cuda_ok(cudaSetDevice(*selected_ordinal), "cudaSetDevice")) {
        return EXIT_FAILURE;
    }
    cudaDeviceProp selected{};
    if (!cuda_ok(cudaGetDeviceProperties(&selected, *selected_ordinal),
                 "cudaGetDeviceProperties(selected)")) {
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "Selected CUDA device ordinal=%d name=%s cc=%d.%d\n",
                 *selected_ordinal, selected.name, selected.major, selected.minor);
    return EXIT_SUCCESS;
}

}  // namespace tk_sm7x::test
