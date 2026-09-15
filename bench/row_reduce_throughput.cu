#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "row_reduce_reference.cuh"
#include "test_utils.cuh"
#include "tk_sm7x/mma.cuh"
#include "tk_sm7x/ptx_backend.cuh"

namespace {

using tk_sm7x::detail::row_max_op;
using tk_sm7x::detail::row_sum_op;
using tk_sm7x::test::cuda_ok;
using tk_sm7x::test::reference_row_reduce;

constexpr int kTile = 16;
constexpr int kCells = kTile * kTile;
constexpr int kDepth = 16;
constexpr unsigned int kBlocks = 1u << 17;
constexpr int kWarmupIterations = 3;
constexpr int kTimedIterations = 20;

// Every block reads the same operand tile so the measurement is not bounded by
// operand bandwidth, and both kernels of a pair share an identical staging and
// MMA prologue. What differs is only how the row results are produced.
template <class Backend>
__device__ __forceinline__ void prologue(
    const __half* a, const __half* b, typename Backend::accumulator& accumulator,
    __half* as, __half* bs) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    Backend::clear(accumulator);
    for (int linear = lane; linear < kCells; linear += 32) {
        as[linear] = a[linear];
        bs[static_cast<std::size_t>(linear % kTile) * kTile + linear / kTile] = b[linear];
    }
    __syncwarp(0xffffffffu);
    typename Backend::fragment_a af;
    typename Backend::fragment_b bf;
    Backend::load_a(af, as, kTile);
    Backend::load_b(bf, bs, kTile);
    Backend::mma(accumulator, af, bf);
    __syncwarp(0xffffffffu);
}

__device__ __forceinline__ void publish(const float* rows, float* out) {
    __syncwarp(0xffffffffu);
    if (static_cast<int>(threadIdx.x) % 32 == 0) {
        float checksum = 0.0f;
        for (int index = 0; index < 2 * kTile; ++index) checksum += rows[index];
        out[blockIdx.x] = checksum;
    }
}

template <class Backend>
__device__ __forceinline__ void register_body(const __half* a, const __half* b, float* out) {
    __shared__ __align__(32) __half as[kCells];
    __shared__ __align__(32) __half bs[kCells];
    __shared__ __align__(32) float rows[2 * kTile];
    typename Backend::accumulator accumulator;
    prologue<Backend>(a, b, accumulator, as, bs);
    Backend::row_sum(rows, accumulator);
    Backend::row_max(rows + kTile, accumulator);
    publish(rows, out);
}

template <class Backend>
__device__ __forceinline__ void shared_body(const __half* a, const __half* b, float* out) {
    __shared__ __align__(32) __half as[kCells];
    __shared__ __align__(32) __half bs[kCells];
    __shared__ __align__(32) float cs[kCells];
    __shared__ __align__(32) float rows[2 * kTile];
    typename Backend::accumulator accumulator;
    prologue<Backend>(a, b, accumulator, as, bs);
    Backend::store(cs, accumulator, kTile);
    __syncwarp(0xffffffffu);
    reference_row_reduce<row_sum_op>(rows, cs);
    reference_row_reduce<row_max_op>(rows + kTile, cs);
    publish(rows, out);
}

using backend_wmma = tk_sm7x::detail::warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::target>;
using backend_m8 = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm70>;
using backend_m16 = tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm75>;

__global__ void reduce_shared_wmma(const __half* a, const __half* b, float* out) {
    shared_body<backend_wmma>(a, b, out);
}

__global__ void reduce_register_m8(const __half* a, const __half* b, float* out) {
    register_body<backend_m8>(a, b, out);
}

__global__ void reduce_shared_m8(const __half* a, const __half* b, float* out) {
    shared_body<backend_m8>(a, b, out);
}

__global__ void reduce_register_m16(const __half* a, const __half* b, float* out) {
    register_body<backend_m16>(a, b, out);
}

__global__ void reduce_shared_m16(const __half* a, const __half* b, float* out) {
    shared_body<backend_m16>(a, b, out);
}

using kernel_pointer = void (*)(const __half*, const __half*, float*);

struct Measurement {
    const char* name;
    kernel_pointer kernel;
};

bool measure(const Measurement& measurement, const __half* a, const __half* b, float* out,
             double* nanoseconds_per_warp) {
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
    ok = cuda_ok(cudaEventRecord(stop), "cudaEventRecord(stop)") && ok;
    ok = cuda_ok(cudaEventSynchronize(stop), "cudaEventSynchronize") && ok;
    float milliseconds = 0.0f;
    ok = cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime") && ok;
    ok = cuda_ok(cudaEventDestroy(start), "cudaEventDestroy(start)") && ok;
    ok = cuda_ok(cudaEventDestroy(stop), "cudaEventDestroy(stop)") && ok;
    if (!ok) return false;
    const double warps = static_cast<double>(kBlocks) * kTimedIterations;
    *nanoseconds_per_warp = static_cast<double>(milliseconds) * 1.0e6 / warps;
    return true;
}

}  // namespace

int main() {
    int device = -1;
    const int selection = tk_sm7x::test::select_sm75_device(&device);
    if (selection != EXIT_SUCCESS) return selection;

    std::vector<__half> host_a(kCells);
    std::vector<__half> host_b(kCells);
    for (int index = 0; index < kCells; ++index) {
        host_a[index] = __float2half(static_cast<float>(index % 7) * 0.125f);
        host_b[index] = __float2half(static_cast<float>(index % 5) * 0.25f);
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
    prepared = cuda_ok(cudaMemcpy(a, host_a.data(), tile_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(a)") && prepared;
    prepared = cuda_ok(cudaMemcpy(b, host_b.data(), tile_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(b)") && prepared;
    if (!prepared) {
        static_cast<void>(release());
        return EXIT_FAILURE;
    }

    const Measurement measurements[] = {
        {"wmma shared", reduce_shared_wmma},
        {"m8 shared", reduce_shared_m8},
        {"m8 register", reduce_register_m8},
        {"m16 shared", reduce_shared_m16},
        {"m16 register", reduce_register_m16},
    };
    double results[sizeof(measurements) / sizeof(measurements[0])] = {};
    bool measured = true;
    for (std::size_t index = 0; index < sizeof(results) / sizeof(results[0]); ++index) {
        measured = measure(measurements[index], a, b, out, &results[index]) && measured;
        if (!measured) break;
        std::printf("%-14s %10.4f ns/warp\n", measurements[index].name, results[index]);
    }
    if (measured) {
        std::printf("m8 shared/register ratio %.4f\n", results[1] / results[2]);
        std::printf("m16 shared/register ratio %.4f\n", results[3] / results[4]);
        std::printf("row reduce throughput: %u warps x %d iterations, K=%d, one warp per block, "
                    "identical prologue\n", kBlocks, kTimedIterations, kDepth);
    }
    return release() && measured ? EXIT_SUCCESS : EXIT_FAILURE;
}
