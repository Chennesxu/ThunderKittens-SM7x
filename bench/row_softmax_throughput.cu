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
constexpr double kProbabilityBound = 512.0 * 0x1p-23;
constexpr int kWarmupIterations = 2;
constexpr int kTimedIterations = 8;
constexpr int kRounds = 4;
constexpr int kMeasurementCount = 6;

using tk_sm7x::test::cuda_ok;

int numerator_a(int row, int col) {
    return (row * 3 + col * 5) % 13 - 6;
}

int numerator_b(int row, int col) {
    return (row * 7 + col * 2) % 11 - 5;
}

std::array<double, kCells> reference_probabilities() {
    std::array<double, kCells> probabilities{};
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
            probabilities[static_cast<std::size_t>(row) * kTile + col] =
                exponent[col] / denominator;
        }
    }
    return probabilities;
}

double expected_checksum(const std::array<double, kCells>& probabilities) {
    double checksum = 0.0;
    for (int row = 0; row < kTile; ++row) {
        for (int col = 0; col < kTile; ++col) {
            const int weight = 1 + ((row * 7 + col * 11) % 31);
            checksum += weight * probabilities[static_cast<std::size_t>(row) * kTile + col];
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

__device__ __forceinline__ void scalar_softmax(float* probabilities, const float* scratch) {
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
}

template <class Op>
__device__ __forceinline__ float balanced_half_reduce(const float* values) {
    const float low = Op::apply(Op::apply(values[0], values[1]),
                                Op::apply(values[2], values[3]));
    const float high = Op::apply(Op::apply(values[4], values[5]),
                                 Op::apply(values[6], values[7]));
    return Op::apply(low, high);
}

__device__ __forceinline__ void two_lane_softmax(float* probabilities, const float* scratch) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int row = lane / 2;
    const int half = lane % 2;
    const std::size_t offset = static_cast<std::size_t>(row) * kTile + half * 8;
    const float* values = scratch + offset;
    const float half_max = balanced_half_reduce<tk_sm7x::detail::row_max_op>(values);
    const float other_max = __shfl_xor_sync(0xffffffffu, half_max, 1);
    const float maximum = half == 0
        ? tk_sm7x::detail::row_max_op::apply(half_max, other_max)
        : tk_sm7x::detail::row_max_op::apply(other_max, half_max);
    float exponent[8];
#pragma unroll
    for (int col = 0; col < 8; ++col) exponent[col] = expf(values[col] - maximum);
    const float half_sum = balanced_half_reduce<tk_sm7x::detail::row_sum_op>(exponent);
    const float other_sum = __shfl_xor_sync(0xffffffffu, half_sum, 1);
    const float denominator = half == 0 ? half_sum + other_sum : other_sum + half_sum;
#pragma unroll
    for (int col = 0; col < 8; ++col) {
        probabilities[offset + col] = exponent[col] / denominator;
    }
    __syncwarp(0xffffffffu);
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
    scalar_softmax(probabilities, scratch);
    publish(probabilities, out);
}

using backend_wmma = tk_sm7x::detail::warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::target>;
using backend_m8 = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm70>;
using backend_m16 = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm75>;

__global__ void softmax_shared_wmma(const __half* a, const __half* b, float* out) {
    shared_body<backend_wmma>(a, b, out);
}

__global__ void softmax_shared_wmma_two_lane(const __half* a, const __half* b, float* out) {
    __shared__ __align__(32) __half shared_a[kCells];
    __shared__ __align__(32) __half shared_b[kCells];
    __shared__ __align__(32) float probabilities[kCells];
    __shared__ __align__(32) float scratch[kCells];
    backend_wmma::accumulator accumulator;
    prologue<backend_wmma>(a, b, accumulator, shared_a, shared_b);
    backend_wmma::store(scratch, accumulator, kTile);
    __syncwarp(0xffffffffu);
    two_lane_softmax(probabilities, scratch);
    publish(probabilities, out);
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

__global__ void softmax_values_shared_wmma(const __half* a, const __half* b, float* values) {
    __shared__ __align__(32) __half shared_a[kCells];
    __shared__ __align__(32) __half shared_b[kCells];
    __shared__ __align__(32) float probabilities[kCells];
    __shared__ __align__(32) float scratch[kCells];
    backend_wmma::accumulator accumulator;
    prologue<backend_wmma>(a, b, accumulator, shared_a, shared_b);
    backend_wmma::store(scratch, accumulator, kTile);
    __syncwarp(0xffffffffu);
    scalar_softmax(probabilities, scratch);
    const int lane = static_cast<int>(threadIdx.x) % 32;
    for (int index = lane; index < kCells; index += 32) values[index] = probabilities[index];
}

__global__ void softmax_values_two_lane(const __half* a, const __half* b, float* values) {
    __shared__ __align__(32) __half shared_a[kCells];
    __shared__ __align__(32) __half shared_b[kCells];
    __shared__ __align__(32) float probabilities[kCells];
    __shared__ __align__(32) float scratch[kCells];
    backend_wmma::accumulator accumulator;
    prologue<backend_wmma>(a, b, accumulator, shared_a, shared_b);
    backend_wmma::store(scratch, accumulator, kTile);
    __syncwarp(0xffffffffu);
    two_lane_softmax(probabilities, scratch);
    const int lane = static_cast<int>(threadIdx.x) % 32;
    for (int index = lane; index < kCells; index += 32) values[index] = probabilities[index];
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

bool validate_two_lane(const __half* a, const __half* b,
                       const std::array<double, kCells>& reference) {
    float* scalar = nullptr;
    float* candidate = nullptr;
    const auto release = [&]() {
        bool ok = true;
        if (scalar != nullptr) ok = cuda_ok(cudaFree(scalar), "cudaFree(scalar)") && ok;
        if (candidate != nullptr) ok = cuda_ok(cudaFree(candidate), "cudaFree(candidate)") && ok;
        return ok;
    };
    const std::size_t bytes = kCells * sizeof(float);
    bool ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scalar), bytes),
                      "cudaMalloc(scalar)") &&
              cuda_ok(cudaMalloc(reinterpret_cast<void**>(&candidate), bytes),
                      "cudaMalloc(candidate)");
    if (ok) {
        ok = cuda_ok(cudaMemset(scalar, 0xff, bytes), "cudaMemset(scalar)") && ok;
        ok = cuda_ok(cudaMemset(candidate, 0xff, bytes), "cudaMemset(candidate)") && ok;
        softmax_values_shared_wmma<<<1, 32>>>(a, b, scalar);
        softmax_values_two_lane<<<1, 32>>>(a, b, candidate);
        ok = cuda_ok(cudaGetLastError(), "value launches") && ok;
        ok = cuda_ok(cudaDeviceSynchronize(), "value synchronize") && ok;
    }
    std::array<float, kCells> host_scalar{};
    std::array<float, kCells> host_candidate{};
    if (ok) {
        ok = cuda_ok(cudaMemcpy(host_scalar.data(), scalar, bytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy(scalar)") && ok;
        ok = cuda_ok(cudaMemcpy(host_candidate.data(), candidate, bytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy(candidate)") && ok;
    }
    if (ok) {
        for (int index = 0; index < kCells; ++index) {
            const double want = reference[index];
            const float control = host_scalar[index];
            const float got = host_candidate[index];
            if (!std::isfinite(control) ||
                std::fabs(static_cast<double>(control) - want) > kProbabilityBound) {
                std::fprintf(stderr, "WMMA scalar cell=%d got=%.17g want=%.17g\n",
                             index, static_cast<double>(control), want);
                ok = false;
                break;
            }
            if (!std::isfinite(got) ||
                std::fabs(static_cast<double>(got) - want) > kProbabilityBound ||
                got != control) {
                std::fprintf(stderr, "WMMA two-lane cell=%d got=%.17g scalar=%.17g want=%.17g\n",
                             index, static_cast<double>(got), static_cast<double>(control), want);
                ok = false;
                break;
            }
        }
    }
    if (ok) std::printf("WMMA two-lane 256/256: PASS exact-scalar and CPU-bound\n");
    const bool released = release();
    return ok && released;
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
        {"wmma two-lane", softmax_shared_wmma_two_lane},
        {"m8 shared", softmax_shared_m8},
        {"m8 register", softmax_register_m8},
        {"m16 shared", softmax_shared_m16},
        {"m16 register", softmax_register_m16},
    };
    const auto reference = reference_probabilities();
    if (!validate_two_lane(a, b, reference)) {
        static_cast<void>(release());
        return EXIT_FAILURE;
    }
    const double expected = expected_checksum(reference);
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
        std::printf("wmma shared/two-lane median ratio %.4f\n", median[0] / median[1]);
        std::printf("m8 shared/register median ratio %.4f\n", median[2] / median[3]);
        std::printf("m16 shared/register median ratio %.4f\n", median[4] / median[5]);
        std::printf("row softmax throughput: %u warps x %d iterations x %d rounds, "
                    "K=%d, one warp per block, identical backend-pair prologue\n",
                    kBlocks, kTimedIterations, kRounds, kTile);
    }
    return release() && ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
