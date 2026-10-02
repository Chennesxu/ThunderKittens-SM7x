#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "test_utils.cuh"
#include "tk_sm7x/mma.cuh"
#include "tk_sm7x/ptx_backend.cuh"
#include "tk_sm7x/row_reduce_ops.cuh"

namespace {

constexpr int kTile = 16;
constexpr int kCells = kTile * kTile;
constexpr unsigned int kBlocks = 1u << 14;
constexpr double kChecksumBound = 0.05;
constexpr int kWarmupIterations = 2;
constexpr int kTimedIterations = 8;
constexpr int kRounds = 4;
constexpr int kMeasurementCount = 5;

using tk_sm7x::test::cuda_ok;

int numerator_a(int row, int col) {
    return (row * 3 + col * 5) % 13 - 6;
}

int numerator_b(int row, int col) {
    return (row * 7 + col * 2) % 11 - 5;
}

double expected_checksum() {
    double checksum = 0.0;
    for (int row = 0; row < kTile; ++row) {
        double logits[kTile];
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < kTile; ++col) {
            double value = 0.0;
            for (int k = 0; k < kTile; ++k) {
                value += static_cast<double>(numerator_a(row, k)) *
                         static_cast<double>(numerator_b(k, col)) / 64.0;
            }
            logits[col] = value;
            if (value > maximum) maximum = value;
        }
        double exponent[kTile];
        double denominator = 0.0;
        for (int col = 0; col < kTile; ++col) {
            exponent[col] = std::exp(logits[col] - maximum);
            denominator += exponent[col];
        }
        for (int col = 0; col < kTile; ++col) {
            const int weight = 1 + ((row * 7 + col * 11) % 31);
            checksum += weight * exponent[col] / denominator;
        }
    }
    return checksum;
}

template <class Backend>
__device__ __forceinline__ void prologue(
    const __half* a, const __half* b, typename Backend::accumulator& accumulator,
    __half* shared_a, __half* shared_b) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    Backend::clear(accumulator);
    for (int linear = lane; linear < kCells; linear += 32) {
        shared_a[linear] = a[linear];
        shared_b[static_cast<std::size_t>(linear % kTile) * kTile + linear / kTile] = b[linear];
    }
    __syncwarp(0xffffffffu);
    typename Backend::fragment_a af;
    typename Backend::fragment_b bf;
    Backend::load_a(af, shared_a, kTile);
    Backend::load_b(bf, shared_b, kTile);
    Backend::mma(accumulator, af, bf);
    __syncwarp(0xffffffffu);
}

__device__ __forceinline__ void publish(const float* probabilities, float* out) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    float checksum = 0.0f;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        const int linear = lane + group * 32;
        const int row = linear / 16;
        const int col = linear % 16;
        const int weight = 1 + ((row * 7 + col * 11) % 31);
        checksum += weight * probabilities[linear];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        checksum += __shfl_down_sync(0xffffffffu, checksum, offset);
    }
    if (lane == 0) out[blockIdx.x] = checksum;
}

template <class Backend>
__device__ __forceinline__ void register_body(const __half* a, const __half* b, float* out) {
    __shared__ __align__(32) __half shared_a[kCells];
    __shared__ __align__(32) __half shared_b[kCells];
    __shared__ __align__(32) float probabilities[kCells];
    __shared__ __align__(32) float scratch[kCells];
    typename Backend::accumulator accumulator;
    prologue<Backend>(a, b, accumulator, shared_a, shared_b);
    Backend::softmax(probabilities, accumulator, scratch);
    publish(probabilities, out);
}

template <class Backend>
__device__ __forceinline__ void shared_body(const __half* a, const __half* b, float* out) {
    __shared__ __align__(32) __half shared_a[kCells];
    __shared__ __align__(32) __half shared_b[kCells];
    __shared__ __align__(32) float probabilities[kCells];
    __shared__ __align__(32) float scratch[kCells];
    typename Backend::accumulator accumulator;
    prologue<Backend>(a, b, accumulator, shared_a, shared_b);
    Backend::store(scratch, accumulator, kTile);
    __syncwarp(0xffffffffu);
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (lane < kTile) {
        const std::size_t row_offset = static_cast<std::size_t>(lane) * kTile;
        const float* row = scratch + row_offset;
        const float maximum =
            tk_sm7x::detail::balanced_row_reduce<tk_sm7x::detail::row_max_op>(row);
        float exponent[kTile];
#pragma unroll
        for (int col = 0; col < kTile; ++col) exponent[col] = expf(row[col] - maximum);
        const float denominator =
            tk_sm7x::detail::balanced_row_reduce<tk_sm7x::detail::row_sum_op>(exponent);
#pragma unroll
        for (int col = 0; col < kTile; ++col) {
            probabilities[row_offset + col] = exponent[col] / denominator;
        }
    }
    __syncwarp(0xffffffffu);
    publish(probabilities, out);
}

using backend_wmma = tk_sm7x::detail::warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::target>;
using backend_m8 = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm70>;
using backend_m16 = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm75>;

__global__ void softmax_shared_wmma(const __half* a, const __half* b, float* out) {
    shared_body<backend_wmma>(a, b, out);
}

__global__ void softmax_shared_m8(const __half* a, const __half* b, float* out) {
    shared_body<backend_m8>(a, b, out);
}

__global__ void softmax_register_m8(const __half* a, const __half* b, float* out) {
    register_body<backend_m8>(a, b, out);
}

__global__ void softmax_shared_m16(const __half* a, const __half* b, float* out) {
    shared_body<backend_m16>(a, b, out);
}

__global__ void softmax_register_m16(const __half* a, const __half* b, float* out) {
    register_body<backend_m16>(a, b, out);
}

using kernel_pointer = void (*)(const __half*, const __half*, float*);

struct Measurement {
    const char* name;
    kernel_pointer kernel;
};

bool measure(const Measurement& measurement, const __half* a, const __half* b,
             float* out, double* nanoseconds_per_warp) {
    for (int i = 0; i < kWarmupIterations; ++i) {
        measurement.kernel<<<kBlocks, 32>>>(a, b, out);
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "warmup synchronize")) return false;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "cudaEventCreate(start)")) return false;
    if (!cuda_ok(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        static_cast<void>(cuda_ok(cudaEventDestroy(start), "cudaEventDestroy(start)"));
        return false;
    }
    bool ok = cuda_ok(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < kTimedIterations; ++i) {
        measurement.kernel<<<kBlocks, 32>>>(a, b, out);
    }
    ok = cuda_ok(cudaGetLastError(), "timed launch") && ok;
    ok = cuda_ok(cudaEventRecord(stop), "cudaEventRecord(stop)") && ok;
    ok = cuda_ok(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)") && ok;
    float milliseconds = 0.0f;
    if (ok) {
        ok = cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop),
                     "cudaEventElapsedTime") && ok;
    }
    ok = cuda_ok(cudaEventDestroy(start), "cudaEventDestroy(start)") && ok;
    ok = cuda_ok(cudaEventDestroy(stop), "cudaEventDestroy(stop)") && ok;
    if (!ok) return false;
    *nanoseconds_per_warp =
        static_cast<double>(milliseconds) * 1.0e6 / (kBlocks * kTimedIterations);
    return std::isfinite(*nanoseconds_per_warp) && *nanoseconds_per_warp > 0.0;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selected = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selected != EXIT_SUCCESS) return selected;

    std::vector<__half> host_a(kCells);
    std::vector<__half> host_b(kCells);
    for (int row = 0; row < kTile; ++row) {
        for (int col = 0; col < kTile; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * kTile + col;
            host_a[index] = __float2half(static_cast<float>(numerator_a(row, col)) / 8.0f);
            host_b[index] = __float2half(static_cast<float>(numerator_b(row, col)) / 8.0f);
        }
    }

    __half* a = nullptr;
    __half* b = nullptr;
    float* out = nullptr;
    const auto release = [&]() {
        bool ok = true;
        if (a != nullptr) ok = cuda_ok(cudaFree(a), "cudaFree(a)") && ok;
        if (b != nullptr) ok = cuda_ok(cudaFree(b), "cudaFree(b)") && ok;
        if (out != nullptr) ok = cuda_ok(cudaFree(out), "cudaFree(out)") && ok;
        return ok;
    };
    const std::size_t tile_bytes = kCells * sizeof(__half);
    bool prepared = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&a), tile_bytes), "cudaMalloc(a)") &&
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b), tile_bytes), "cudaMalloc(b)") &&
                    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), kBlocks * sizeof(float)),
                            "cudaMalloc(out)");
    if (!prepared) {
        static_cast<void>(release());
        return EXIT_FAILURE;
    }
    prepared = cuda_ok(cudaMemcpy(a, host_a.data(), tile_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(a)") && prepared;
    prepared = cuda_ok(cudaMemcpy(b, host_b.data(), tile_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(b)") && prepared;
    if (!prepared) {
        static_cast<void>(release());
        return EXIT_FAILURE;
    }

    const Measurement measurements[] = {
        {"wmma shared", softmax_shared_wmma},
        {"m8 shared", softmax_shared_m8},
        {"m8 register", softmax_register_m8},
        {"m16 shared", softmax_shared_m16},
        {"m16 register", softmax_register_m16},
    };
    const double expected = expected_checksum();
    bool ok = true;
    for (const auto& measurement : measurements) {
        ok = cuda_ok(cudaMemset(out, 0xff, kBlocks * sizeof(float)),
                     "cudaMemset(checksum)") && ok;
        if (!ok) break;
        measurement.kernel<<<kBlocks, 32>>>(a, b, out);
        ok = cuda_ok(cudaGetLastError(), "checksum launch") && ok;
        ok = cuda_ok(cudaDeviceSynchronize(), "checksum synchronize") && ok;
        const unsigned int sample_blocks[] = {0, kBlocks / 2, kBlocks - 1};
        for (unsigned int block : sample_blocks) {
            float sample = 0.0f;
            ok = cuda_ok(cudaMemcpy(&sample, out + block, sizeof(float), cudaMemcpyDeviceToHost),
                         "cudaMemcpy(checksum)") && ok;
            if (!ok || !std::isfinite(sample) ||
                std::fabs(static_cast<double>(sample) - expected) > kChecksumBound) {
                std::fprintf(stderr, "%s block=%u checksum got=%.17g expected=%.17g\n",
                             measurement.name, block, static_cast<double>(sample), expected);
                ok = false;
                break;
            }
        }
        if (!ok) break;
        std::printf("%s checksum: PASS expected=%.17g\n", measurement.name, expected);
    }
    double samples[kMeasurementCount][kRounds] = {};
    for (int round = 0; ok && round < kRounds; ++round) {
        for (int position = 0; position < kMeasurementCount; ++position) {
            const int index = round % 2 == 0 ? position : kMeasurementCount - 1 - position;
            ok = measure(measurements[index], a, b, out, &samples[index][round]);
            if (!ok) break;
        }
    }
    if (ok) {
        double median[kMeasurementCount] = {};
        for (int index = 0; index < kMeasurementCount; ++index) {
            std::array<double, kRounds> sorted{};
            for (int round = 0; round < kRounds; ++round) sorted[round] = samples[index][round];
            std::sort(sorted.begin(), sorted.end());
            median[index] = (sorted[1] + sorted[2]) / 2.0;
            std::printf("%-14s median=%8.4f range=[%8.4f,%8.4f] amortized ns/warp\n",
                        measurements[index].name, median[index], sorted.front(), sorted.back());
        }
        std::printf("m8 shared/register median ratio %.4f\n", median[1] / median[2]);
        std::printf("m16 shared/register median ratio %.4f\n", median[3] / median[4]);
        std::printf("row softmax throughput: %u warps x %d iterations x %d rounds, "
                    "K=%d, one warp per block, identical backend-pair prologue\n",
                    kBlocks, kTimedIterations, kRounds, kTile);
    }
    return release() && ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
