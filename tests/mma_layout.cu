#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "test_utils.cuh"
#include "mma_layout_oracle.cuh"
#include "tk_sm7x/ptx_mma.cuh"

namespace {

using tk_sm7x::test::cuda_ok;

// TK_SM7X_LAYOUT_PERTURB exists so the suite can prove this test detects a wrong
// layout. It is never defined by the build.
#ifndef TK_SM7X_LAYOUT_PERTURB
#define TK_SM7X_LAYOUT_PERTURB 0
#endif

constexpr float kTolerance = 1e-3f;

// Outputs start as a quiet NaN so a position the kernel never writes fails the
// finiteness check instead of coinciding with a legitimate zero result.
const float kUnwritten = std::nanf("");

using tk_sm7x::test::kM8AccumulatorCell;
using tk_sm7x::test::kM8OperandCell;
using tk_sm7x::test::kWarpLanes;
#if defined(KITTENS_SM75)
using tk_sm7x::test::kM16FragmentCell;
using tk_sm7x::test::kM16OperandBCell;
#endif

bool check_m8n8k4_oracle() {
    using mma = tk_sm7x::detail::mma_m8n8k4;
    for (int lane = 0; lane < kWarpLanes; ++lane) {
        const int quadpair = mma::quadpair(lane);
        const int row = quadpair * mma::kTile + mma::position(lane);
        for (int reg = 0; reg < mma::kOperandRegisters; ++reg) {
            for (int half = 0; half < 2; ++half) {
                const int operand = row * mma::kDepth + mma::operand_slot(reg, half);
                if (operand != kM8OperandCell[lane][reg * 2 + half]) {
                    std::fprintf(stderr,
                                 "m8n8k4 oracle: lane %d register %d half %d cell %d, "
                                 "table %d\n",
                                 lane, reg, half, operand,
                                 kM8OperandCell[lane][reg * 2 + half]);
                    return false;
                }
            }
        }

        int cell[mma::kAccumulators];
        for (int slot = 0; slot < mma::kAccumulators; ++slot) {
            cell[slot] = -1;
        }
        for (int row = 0; row < mma::kTile; ++row) {
            for (int col = 0; col < mma::kTile; ++col) {
                if (mma::accumulator_lane(quadpair, row, col) != lane) {
                    continue;
                }
                const int slot = mma::accumulator_index(row, col);
                if (slot < 0 || slot >= mma::kAccumulators || cell[slot] != -1) {
                    std::fprintf(stderr, "m8n8k4 oracle: lane %d slot %d is not unique\n",
                                 lane, slot);
                    return false;
                }
                cell[slot] = row * mma::kTile + col;
            }
        }
        for (int slot = 0; slot < mma::kAccumulators; ++slot) {
            if (cell[slot] != kM8AccumulatorCell[lane][slot]) {
                std::fprintf(stderr,
                             "m8n8k4 oracle: lane %d slot %d cell %d, table %d\n",
                             lane, slot, cell[slot], kM8AccumulatorCell[lane][slot]);
                return false;
            }
        }
    }
    std::printf("m8n8k4 oracle: PASS\n");
    return true;
}

#if defined(KITTENS_SM75)
bool check_m16n8k8_oracle() {
    using mma = tk_sm7x::detail::mma_m16n8k8;
    for (int lane = 0; lane < kWarpLanes; ++lane) {
        for (int reg = 0; reg < mma::kOperandARegisters; ++reg) {
            for (int half = 0; half < 2; ++half) {
                const int slot = mma::operand_slot(reg, half);
                const int a_cell = mma::operand_a_row(lane, slot) * mma::kDepth +
                                   mma::operand_a_col(lane, slot);
                if (a_cell != kM16FragmentCell[lane][reg * 2 + half]) {
                    std::fprintf(stderr,
                                 "m16n8k8 oracle: lane %d A register %d half %d cell %d, "
                                 "table %d\n",
                                 lane, reg, half, a_cell,
                                 kM16FragmentCell[lane][reg * 2 + half]);
                    return false;
                }
            }
        }
        for (int half = 0; half < 2; ++half) {
            const int slot = mma::operand_slot(0, half);
            const int b_cell =
                mma::operand_b_row(lane, slot) * mma::kCols + mma::operand_b_col(lane);
            if (b_cell != kM16OperandBCell[lane][half]) {
                std::fprintf(stderr,
                             "m16n8k8 oracle: lane %d B half %d cell %d, table %d\n",
                             lane, half, b_cell, kM16OperandBCell[lane][half]);
                return false;
            }
        }
        for (int i = 0; i < mma::kAccumulators; ++i) {
            const int c_cell = mma::accumulator_row(lane, i) * mma::kCols +
                               mma::accumulator_col(lane, i);
            if (c_cell != kM16FragmentCell[lane][i]) {
                std::fprintf(stderr, "m16n8k8 oracle: lane %d C slot %d cell %d, table %d\n",
                             lane, i, c_cell, kM16FragmentCell[lane][i]);
                return false;
            }
        }
    }
    std::printf("m16n8k8 oracle: PASS\n");
    return true;
}
#endif

__global__ void run_m8n8k4(const __half* a, const __half* b, const float* c, float* d) {
    using mma = tk_sm7x::detail::mma_m8n8k4;
    const int lane = static_cast<int>(threadIdx.x);
    const int quadpair = mma::quadpair(lane);
    const int position = mma::position(lane);

    const std::size_t operand_row =
        static_cast<std::size_t>(quadpair) * mma::kTile + static_cast<std::size_t>(position);
    uint32_t operand_a[mma::kOperandRegisters];
    uint32_t operand_b[mma::kOperandRegisters];
    for (int reg = 0; reg < mma::kOperandRegisters; ++reg) {
        __half half_a[2];
        __half half_b[2];
        for (int half = 0; half < 2; ++half) {
            const std::size_t index =
                operand_row * mma::kDepth +
                static_cast<std::size_t>(mma::operand_slot(reg, half));
            half_a[half] = a[index];
            half_b[half] = b[index];
        }
        operand_a[reg] = tk_sm7x::detail::pack_halves(half_a[0], half_a[1]);
        operand_b[reg] = tk_sm7x::detail::pack_halves(half_b[0], half_b[1]);
    }

    float accumulator[8];
    for (int q = 0; q < mma::kQuadpairs; ++q) {
        for (int row = 0; row < mma::kTile; ++row) {
            for (int col = 0; col < mma::kTile; ++col) {
                if (mma::accumulator_lane(q, row, col) != lane) {
                    continue;
                }
#if TK_SM7X_LAYOUT_PERTURB == 5
                const int slot = mma::accumulator_index(row, col) ^ 4;
#else
                const int slot = mma::accumulator_index(row, col);
#endif
                const std::size_t index =
                    (static_cast<std::size_t>(q) * mma::kTile +
                     static_cast<std::size_t>(row)) *
                        mma::kTile +
                    static_cast<std::size_t>(col);
                accumulator[slot] = c[index];
            }
        }
    }
    mma::mma(accumulator, operand_a[0], operand_a[1], operand_b[0], operand_b[1]);

    for (int q = 0; q < mma::kQuadpairs; ++q) {
        for (int row = 0; row < mma::kTile; ++row) {
            for (int col = 0; col < mma::kTile; ++col) {
#if TK_SM7X_LAYOUT_PERTURB == 1
                const int owner = mma::accumulator_lane(q, row, col) ^ 1;
#else
                const int owner = mma::accumulator_lane(q, row, col);
#endif
#if TK_SM7X_LAYOUT_PERTURB == 2
                const int index = mma::accumulator_index(col, row);
#else
                const int index = mma::accumulator_index(row, col);
#endif
                if (owner == lane) {
                    const std::size_t cell =
                        (static_cast<std::size_t>(q) * mma::kTile +
                         static_cast<std::size_t>(row)) *
                            mma::kTile +
                        static_cast<std::size_t>(col);
                    d[cell] = accumulator[index];
                }
            }
        }
    }
}

bool check_m8n8k4(int ordinal) {
    using mma = tk_sm7x::detail::mma_m8n8k4;
    constexpr int kOperands = mma::kQuadpairs * mma::kTile * mma::kDepth;
    constexpr int kOutputs = mma::kQuadpairs * mma::kTile * mma::kTile;

    std::vector<__half> host_a(kOperands);
    std::vector<__half> host_b(kOperands);
    unsigned int state = 0x5eed0088u;
    auto next = [&state] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 16) % 13u) - 6.0f;
    };
    for (int i = 0; i < kOperands; ++i) {
        host_a[i] = __float2half(next());
        host_b[i] = __float2half(next());
    }

    // Position-recoverable accumulator input, so a wrong C layout cannot cancel.
    std::vector<float> host_c(kOutputs);
    for (int q = 0; q < mma::kQuadpairs; ++q) {
        for (int row = 0; row < mma::kTile; ++row) {
            for (int col = 0; col < mma::kTile; ++col) {
                host_c[(q * mma::kTile + row) * mma::kTile + col] =
                    static_cast<float>(1000 * q + 100 * row + 10 * col + 1);
            }
        }
    }

    std::vector<float> expected(kOutputs, 0.0f);
    for (int q = 0; q < mma::kQuadpairs; ++q) {
        for (int row = 0; row < mma::kTile; ++row) {
            for (int col = 0; col < mma::kTile; ++col) {
                float acc = host_c[(q * mma::kTile + row) * mma::kTile + col];
                for (int k = 0; k < mma::kDepth; ++k) {
                    acc += __half2float(host_a[(q * mma::kTile + row) * mma::kDepth + k]) *
                           __half2float(host_b[(q * mma::kTile + col) * mma::kDepth + k]);
                }
                expected[(q * mma::kTile + row) * mma::kTile + col] = acc;
            }
        }
    }

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_c = nullptr;
    float* device_d = nullptr;
    bool ok = cuda_ok(cudaMalloc(&device_a, kOperands * sizeof(__half)), "cudaMalloc(a)") &&
              cuda_ok(cudaMalloc(&device_b, kOperands * sizeof(__half)), "cudaMalloc(b)") &&
              cuda_ok(cudaMalloc(&device_c, kOutputs * sizeof(float)), "cudaMalloc(c)") &&
              cuda_ok(cudaMalloc(&device_d, kOutputs * sizeof(float)), "cudaMalloc(d)");
    if (ok) {
        ok = cuda_ok(cudaMemcpy(device_a, host_a.data(), kOperands * sizeof(__half),
                                cudaMemcpyHostToDevice), "cudaMemcpy(a)") &&
             cuda_ok(cudaMemcpy(device_b, host_b.data(), kOperands * sizeof(__half),
                                cudaMemcpyHostToDevice), "cudaMemcpy(b)") &&
             cuda_ok(cudaMemcpy(device_c, host_c.data(), kOutputs * sizeof(float),
                                cudaMemcpyHostToDevice), "cudaMemcpy(c)");
    }
    std::vector<float> sentinel(kOutputs, kUnwritten);
    if (ok) {
        ok = cuda_ok(cudaMemcpy(device_d, sentinel.data(), kOutputs * sizeof(float),
                                cudaMemcpyHostToDevice), "seed d");
    }
    if (ok) {
        run_m8n8k4<<<1, kWarpLanes>>>(device_a, device_b, device_c, device_d);
        ok = cuda_ok(cudaGetLastError(), "m8n8k4 launch") &&
             cuda_ok(cudaDeviceSynchronize(), "m8n8k4 synchronize");
    }
    std::vector<float> observed(kOutputs, 0.0f);
    if (ok) {
        ok = cuda_ok(cudaMemcpy(observed.data(), device_d, kOutputs * sizeof(float),
                                cudaMemcpyDeviceToHost), "cudaMemcpy(d)");
    }
    if (ok) {
        for (int i = 0; i < kOutputs; ++i) {
            if (!std::isfinite(observed[i])) {
                std::fprintf(stderr, "m8n8k4 output %d was never written\n", i);
                ok = false;
                break;
            }
            if (std::fabs(observed[i] - expected[i]) > kTolerance) {
                std::fprintf(stderr, "m8n8k4 layout mismatch at %d: got %f want %f\n", i,
                             static_cast<double>(observed[i]),
                             static_cast<double>(expected[i]));
                ok = false;
                break;
            }
        }
    }

    bool released = cuda_ok(cudaFree(device_a), "cudaFree(a)");
    released = cuda_ok(cudaFree(device_b), "cudaFree(b)") && released;
    released = cuda_ok(cudaFree(device_c), "cudaFree(c)") && released;
    released = cuda_ok(cudaFree(device_d), "cudaFree(d)") && released;
    if (ok && released) {
        std::printf("m8n8k4 layout: PASS ordinal=%d\n", ordinal);
    }
    return ok && released;
}

#if defined(KITTENS_SM75)
__global__ void run_m16n8k8(const __half* a, const __half* b, const float* c, float* d) {
    using mma = tk_sm7x::detail::mma_m16n8k8;
    const int lane = static_cast<int>(threadIdx.x);

    uint32_t operand_a[mma::kOperandARegisters];
    for (int reg = 0; reg < mma::kOperandARegisters; ++reg) {
        __half half_a[2];
        for (int half = 0; half < 2; ++half) {
            const int slot = mma::operand_slot(reg, half);
            const std::size_t index =
                static_cast<std::size_t>(mma::operand_a_row(lane, slot)) * mma::kDepth +
                static_cast<std::size_t>(mma::operand_a_col(lane, slot));
            half_a[half] = a[index];
        }
        operand_a[reg] = tk_sm7x::detail::pack_halves(half_a[0], half_a[1]);
    }
    __half half_b[2];
    for (int half = 0; half < 2; ++half) {
#if TK_SM7X_LAYOUT_PERTURB == 3
        const int col = mma::pair(lane);
#else
        const int col = mma::operand_b_col(lane);
#endif
        const int slot = mma::operand_slot(0, half);
        const std::size_t index =
            static_cast<std::size_t>(mma::operand_b_row(lane, slot)) * mma::kCols +
            static_cast<std::size_t>(col);
        half_b[half] = b[index];
    }
    const uint32_t operand_b = tk_sm7x::detail::pack_halves(half_b[0], half_b[1]);

    float accumulator[4];
    for (int i = 0; i < mma::kAccumulators; ++i) {
#if TK_SM7X_LAYOUT_PERTURB == 6
        const int slot = i ^ 1;
#else
        const int slot = i;
#endif
        const std::size_t index =
            static_cast<std::size_t>(mma::accumulator_row(lane, i)) * mma::kCols +
            static_cast<std::size_t>(mma::accumulator_col(lane, i));
        accumulator[slot] = c[index];
    }
    mma::mma(accumulator, operand_a[0], operand_a[1], operand_b);

    for (int i = 0; i < mma::kAccumulators; ++i) {
#if TK_SM7X_LAYOUT_PERTURB == 4
        const int row = mma::accumulator_row(lane, i) % 8;
#else
        const int row = mma::accumulator_row(lane, i);
#endif
        const std::size_t index = static_cast<std::size_t>(row) * mma::kCols +
                                  static_cast<std::size_t>(mma::accumulator_col(lane, i));
        d[index] = accumulator[i];
    }
}

bool check_m16n8k8(int ordinal) {
    using mma = tk_sm7x::detail::mma_m16n8k8;
    constexpr int kA = mma::kRows * mma::kDepth;
    constexpr int kB = mma::kDepth * mma::kCols;
    constexpr int kD = mma::kRows * mma::kCols;

    std::vector<__half> host_a(kA);
    std::vector<__half> host_b(kB);
    unsigned int state = 0x1337beefu;
    auto next = [&state] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 16) % 17u) - 8.0f;
    };
    for (int i = 0; i < kA; ++i) {
        host_a[i] = __float2half(next());
    }
    for (int i = 0; i < kB; ++i) {
        host_b[i] = __float2half(next());
    }

    std::vector<float> host_c(kD);
    for (int row = 0; row < mma::kRows; ++row) {
        for (int col = 0; col < mma::kCols; ++col) {
            host_c[row * mma::kCols + col] =
                static_cast<float>(100 * row + 10 * col + 1);
        }
    }

    std::vector<float> expected(kD, 0.0f);
    for (int row = 0; row < mma::kRows; ++row) {
        for (int col = 0; col < mma::kCols; ++col) {
            float acc = host_c[row * mma::kCols + col];
            for (int k = 0; k < mma::kDepth; ++k) {
                acc += __half2float(host_a[row * mma::kDepth + k]) *
                       __half2float(host_b[k * mma::kCols + col]);
            }
            expected[row * mma::kCols + col] = acc;
        }
    }

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_c = nullptr;
    float* device_d = nullptr;
    bool ok = cuda_ok(cudaMalloc(&device_a, kA * sizeof(__half)), "cudaMalloc(a)") &&
              cuda_ok(cudaMalloc(&device_b, kB * sizeof(__half)), "cudaMalloc(b)") &&
              cuda_ok(cudaMalloc(&device_c, kD * sizeof(float)), "cudaMalloc(c)") &&
              cuda_ok(cudaMalloc(&device_d, kD * sizeof(float)), "cudaMalloc(d)");
    if (ok) {
        ok = cuda_ok(cudaMemcpy(device_a, host_a.data(), kA * sizeof(__half),
                                cudaMemcpyHostToDevice), "cudaMemcpy(a)") &&
             cuda_ok(cudaMemcpy(device_b, host_b.data(), kB * sizeof(__half),
                                cudaMemcpyHostToDevice), "cudaMemcpy(b)") &&
             cuda_ok(cudaMemcpy(device_c, host_c.data(), kD * sizeof(float),
                                cudaMemcpyHostToDevice), "cudaMemcpy(c)");
    }
    std::vector<float> sentinel(kD, kUnwritten);
    if (ok) {
        ok = cuda_ok(cudaMemcpy(device_d, sentinel.data(), kD * sizeof(float),
                                cudaMemcpyHostToDevice), "seed d");
    }
    if (ok) {
        run_m16n8k8<<<1, kWarpLanes>>>(device_a, device_b, device_c, device_d);
        ok = cuda_ok(cudaGetLastError(), "m16n8k8 launch") &&
             cuda_ok(cudaDeviceSynchronize(), "m16n8k8 synchronize");
    }
    std::vector<float> observed(kD, 0.0f);
    if (ok) {
        ok = cuda_ok(cudaMemcpy(observed.data(), device_d, kD * sizeof(float),
                                cudaMemcpyDeviceToHost), "cudaMemcpy(d)");
    }
    if (ok) {
        for (int i = 0; i < kD; ++i) {
            if (!std::isfinite(observed[i])) {
                std::fprintf(stderr, "m16n8k8 output %d was never written\n", i);
                ok = false;
                break;
            }
            if (std::fabs(observed[i] - expected[i]) > kTolerance) {
                std::fprintf(stderr, "m16n8k8 layout mismatch at %d: got %f want %f\n", i,
                             static_cast<double>(observed[i]),
                             static_cast<double>(expected[i]));
                ok = false;
                break;
            }
        }
    }

    bool released = cuda_ok(cudaFree(device_a), "cudaFree(a)");
    released = cuda_ok(cudaFree(device_b), "cudaFree(b)") && released;
    released = cuda_ok(cudaFree(device_c), "cudaFree(c)") && released;
    released = cuda_ok(cudaFree(device_d), "cudaFree(d)") && released;
    if (ok && released) {
        std::printf("m16n8k8 layout: PASS ordinal=%d\n", ordinal);
    }
    return ok && released;
}
#endif

}  // namespace

int main() {
    bool mapped = check_m8n8k4_oracle();
#if defined(KITTENS_SM75)
    mapped = check_m16n8k8_oracle() && mapped;
#endif
    if (!mapped) {
        return EXIT_FAILURE;
    }

    int ordinal = -1;
    const int selection = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selection != EXIT_SUCCESS) {
        return selection;
    }

    bool ok = check_m8n8k4(ordinal);
#if defined(KITTENS_SM75)
    ok = check_m16n8k8(ordinal) && ok;
#endif
    if (!ok) {
        return EXIT_FAILURE;
    }
    std::printf("mma layout: PASS ordinal=%d\n", ordinal);
    return EXIT_SUCCESS;
}
