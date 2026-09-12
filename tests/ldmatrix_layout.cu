#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mma_layout_oracle.cuh"
#include "test_utils.cuh"
#include "tk_sm7x/ptx_ldmatrix.cuh"

namespace {

using tk_sm7x::test::cuda_ok;
using tk_sm7x::test::kM16FragmentCell;
using tk_sm7x::test::kM16OperandBCell;
using tk_sm7x::test::kWarpLanes;

constexpr int kACells = 16 * 8;
constexpr int kBCells = 8 * 8;
constexpr int kARegisters = kWarpLanes * 2;
constexpr int kBRegisters = kWarpLanes;
constexpr uint32_t kUnwritten = 0xffffffffu;

__global__ void run_ldmatrix_layout(const __half* a, const __half* b,
                                    uint32_t* loaded_a_global,
                                    uint32_t* loaded_b_global) {
    __shared__ __align__(32) __half shared_a[kACells];
    __shared__ __align__(32) __half shared_b[kBCells];

    const int lane = static_cast<int>(threadIdx.x);
    for (int linear = lane; linear < kACells; linear += kWarpLanes) {
        shared_a[linear] = a[linear];
    }
    for (int linear = lane; linear < kBCells; linear += kWarpLanes) {
        const int row = linear / 8;
        const int col = linear % 8;
        shared_b[col * 8 + row] = b[linear];
    }
    __syncthreads();

    uint32_t loaded_a[2];
    const int a_owner = lane % 16;
    tk_sm7x::detail::ldmatrix_x2(
        loaded_a, tk_sm7x::detail::shared_address(shared_a + a_owner * 8));

    uint32_t loaded_b;
    const int b_owner = lane % 8;
    tk_sm7x::detail::ldmatrix_x1(
        loaded_b, tk_sm7x::detail::shared_address(shared_b + b_owner * 8));

    const std::size_t a_base = static_cast<std::size_t>(lane) * 2;
    loaded_a_global[a_base] = loaded_a[0];
    loaded_a_global[a_base + 1] = loaded_a[1];
    loaded_b_global[static_cast<std::size_t>(lane)] = loaded_b;
}

bool check_registers(const std::vector<__half>& host_a,
                     const std::vector<__half>& host_b,
                     const std::vector<uint32_t>& observed_a,
                     const std::vector<uint32_t>& observed_b, int ordinal) {
    bool a_ok = true;
    for (int lane = 0; lane < kWarpLanes && a_ok; ++lane) {
        for (int reg = 0; reg < 2 && a_ok; ++reg) {
            const uint32_t word = observed_a[static_cast<std::size_t>(lane) * 2 + reg];
            if (word == kUnwritten) {
                std::fprintf(stderr, "ldmatrix A lane %d register %d was never written\n",
                             lane, reg);
                a_ok = false;
                break;
            }
            for (int half = 0; half < 2; ++half) {
                const uint16_t bits = static_cast<uint16_t>(word >> (16 * half));
                const int cell = kM16FragmentCell[lane][2 * reg + half];
                const uint16_t expected_bits =
                    static_cast<__half_raw>(host_a[cell]).x;
                if (bits != expected_bits) {
                    std::fprintf(stderr,
                                 "ldmatrix A mismatch lane=%d register=%d half=%d "
                                 "cell=%d got=0x%04x want=0x%04x\n",
                                 lane, reg, half, cell, static_cast<unsigned>(bits),
                                 static_cast<unsigned>(expected_bits));
                    a_ok = false;
                    break;
                }
            }
        }
    }
    if (a_ok) {
        std::printf("ldmatrix A layout: PASS ordinal=%d\n", ordinal);
    }

    bool b_ok = true;
    for (int lane = 0; lane < kWarpLanes && b_ok; ++lane) {
        const uint32_t word = observed_b[lane];
        if (word == kUnwritten) {
            std::fprintf(stderr, "ldmatrix B lane %d was never written\n", lane);
            b_ok = false;
            break;
        }
        for (int half = 0; half < 2; ++half) {
            const uint16_t bits = static_cast<uint16_t>(word >> (16 * half));
            const int cell = kM16OperandBCell[lane][half];
            const uint16_t expected_bits = static_cast<__half_raw>(host_b[cell]).x;
            if (bits != expected_bits) {
                std::fprintf(stderr,
                             "ldmatrix B mismatch lane=%d half=%d cell=%d "
                             "got=0x%04x want=0x%04x\n",
                             lane, half, cell, static_cast<unsigned>(bits),
                             static_cast<unsigned>(expected_bits));
                b_ok = false;
                break;
            }
        }
    }
    if (b_ok) {
        std::printf("ldmatrix B layout: PASS ordinal=%d\n", ordinal);
    }
    return a_ok && b_ok;
}

bool check_ldmatrix_layout(int ordinal) {
    std::vector<__half> host_a(kACells);
    std::vector<__half> host_b(kBCells);
    for (int i = 0; i < kACells; ++i) {
        host_a[i] = __float2half(static_cast<float>(i + 1));
    }
    for (int i = 0; i < kBCells; ++i) {
        host_b[i] = __float2half(static_cast<float>(257 + i));
    }

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    uint32_t* device_loaded_a = nullptr;
    uint32_t* device_loaded_b = nullptr;
    bool ok = cuda_ok(cudaMalloc(&device_a, kACells * sizeof(__half)), "cudaMalloc(a)") &&
              cuda_ok(cudaMalloc(&device_b, kBCells * sizeof(__half)), "cudaMalloc(b)") &&
              cuda_ok(cudaMalloc(&device_loaded_a, kARegisters * sizeof(uint32_t)),
                      "cudaMalloc(loaded_a)") &&
              cuda_ok(cudaMalloc(&device_loaded_b, kBRegisters * sizeof(uint32_t)),
                      "cudaMalloc(loaded_b)");
    if (ok) {
        ok = cuda_ok(cudaMemcpy(device_a, host_a.data(), kACells * sizeof(__half),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(a)") &&
             cuda_ok(cudaMemcpy(device_b, host_b.data(), kBCells * sizeof(__half),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(b)");
    }
    std::vector<uint32_t> sentinel_a(kARegisters, kUnwritten);
    std::vector<uint32_t> sentinel_b(kBRegisters, kUnwritten);
    if (ok) {
        ok = cuda_ok(cudaMemcpy(device_loaded_a, sentinel_a.data(),
                                kARegisters * sizeof(uint32_t), cudaMemcpyHostToDevice),
                     "seed loaded_a") &&
             cuda_ok(cudaMemcpy(device_loaded_b, sentinel_b.data(),
                                kBRegisters * sizeof(uint32_t), cudaMemcpyHostToDevice),
                     "seed loaded_b");
    }
    if (ok) {
        run_ldmatrix_layout<<<1, kWarpLanes>>>(device_a, device_b, device_loaded_a,
                                               device_loaded_b);
        ok = cuda_ok(cudaGetLastError(), "ldmatrix layout launch") &&
             cuda_ok(cudaDeviceSynchronize(), "ldmatrix layout synchronize");
    }

    std::vector<uint32_t> observed_a(kARegisters);
    std::vector<uint32_t> observed_b(kBRegisters);
    if (ok) {
        ok = cuda_ok(cudaMemcpy(observed_a.data(), device_loaded_a,
                                kARegisters * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                     "cudaMemcpy(loaded_a)") &&
             cuda_ok(cudaMemcpy(observed_b.data(), device_loaded_b,
                                kBRegisters * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                     "cudaMemcpy(loaded_b)");
    }
    if (ok) {
        ok = check_registers(host_a, host_b, observed_a, observed_b, ordinal);
    }

    bool released = cuda_ok(cudaFree(device_a), "cudaFree(a)");
    released = cuda_ok(cudaFree(device_b), "cudaFree(b)") && released;
    released = cuda_ok(cudaFree(device_loaded_a), "cudaFree(loaded_a)") && released;
    released = cuda_ok(cudaFree(device_loaded_b), "cudaFree(loaded_b)") && released;
    return ok && released;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selection = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selection != EXIT_SUCCESS) {
        return selection;
    }
    if (!check_ldmatrix_layout(ordinal)) {
        return EXIT_FAILURE;
    }
    std::printf("ldmatrix layout: PASS ordinal=%d\n", ordinal);
    return EXIT_SUCCESS;
}
