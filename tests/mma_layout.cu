#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "test_utils.cuh"
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

// Frozen lane/index oracle for the layout contract. The lane-to-matrix and
// accumulator entries reflect hardware observations; contracted-K slot order is
// the single software convention Phase B staging must satisfy. Literal tables,
// rather than a second derivation from the helpers, reject coherent relabelling.

// mma_m8n8k4 operand cell = (quadpair * kTile + position) * kDepth + k, indexed
// by the packed-register slot the element occupies: [lane][reg * 2 + half].
constexpr int kM8OperandCell[32][4] = {
    {  0,   1,   2,   3},
    {  4,   5,   6,   7},
    {  8,   9,  10,  11},
    { 12,  13,  14,  15},
    { 32,  33,  34,  35},
    { 36,  37,  38,  39},
    { 40,  41,  42,  43},
    { 44,  45,  46,  47},
    { 64,  65,  66,  67},
    { 68,  69,  70,  71},
    { 72,  73,  74,  75},
    { 76,  77,  78,  79},
    { 96,  97,  98,  99},
    {100, 101, 102, 103},
    {104, 105, 106, 107},
    {108, 109, 110, 111},
    { 16,  17,  18,  19},
    { 20,  21,  22,  23},
    { 24,  25,  26,  27},
    { 28,  29,  30,  31},
    { 48,  49,  50,  51},
    { 52,  53,  54,  55},
    { 56,  57,  58,  59},
    { 60,  61,  62,  63},
    { 80,  81,  82,  83},
    { 84,  85,  86,  87},
    { 88,  89,  90,  91},
    { 92,  93,  94,  95},
    {112, 113, 114, 115},
    {116, 117, 118, 119},
    {120, 121, 122, 123},
    {124, 125, 126, 127},
};

// mma_m8n8k4 accumulator cell = row * kTile + col inside the lane's quadpair.
constexpr int kM8AccumulatorCell[32][8] = {
    { 0,  1, 16, 17,  4,  5, 20, 21},
    { 8,  9, 24, 25, 12, 13, 28, 29},
    { 2,  3, 18, 19,  6,  7, 22, 23},
    {10, 11, 26, 27, 14, 15, 30, 31},
    { 0,  1, 16, 17,  4,  5, 20, 21},
    { 8,  9, 24, 25, 12, 13, 28, 29},
    { 2,  3, 18, 19,  6,  7, 22, 23},
    {10, 11, 26, 27, 14, 15, 30, 31},
    { 0,  1, 16, 17,  4,  5, 20, 21},
    { 8,  9, 24, 25, 12, 13, 28, 29},
    { 2,  3, 18, 19,  6,  7, 22, 23},
    {10, 11, 26, 27, 14, 15, 30, 31},
    { 0,  1, 16, 17,  4,  5, 20, 21},
    { 8,  9, 24, 25, 12, 13, 28, 29},
    { 2,  3, 18, 19,  6,  7, 22, 23},
    {10, 11, 26, 27, 14, 15, 30, 31},
    {32, 33, 48, 49, 36, 37, 52, 53},
    {40, 41, 56, 57, 44, 45, 60, 61},
    {34, 35, 50, 51, 38, 39, 54, 55},
    {42, 43, 58, 59, 46, 47, 62, 63},
    {32, 33, 48, 49, 36, 37, 52, 53},
    {40, 41, 56, 57, 44, 45, 60, 61},
    {34, 35, 50, 51, 38, 39, 54, 55},
    {42, 43, 58, 59, 46, 47, 62, 63},
    {32, 33, 48, 49, 36, 37, 52, 53},
    {40, 41, 56, 57, 44, 45, 60, 61},
    {34, 35, 50, 51, 38, 39, 54, 55},
    {42, 43, 58, 59, 46, 47, 62, 63},
    {32, 33, 48, 49, 36, 37, 52, 53},
    {40, 41, 56, 57, 44, 45, 60, 61},
    {34, 35, 50, 51, 38, 39, 54, 55},
    {42, 43, 58, 59, 46, 47, 62, 63},
};

#if defined(KITTENS_SM75)
// mma_m16n8k8 distributes its 16x8 A fragment and its 16x8 accumulator over the
// warp identically, so one table is the oracle for both.
constexpr int kM16FragmentCell[32][4] = {
    {  0,   1,  64,  65},
    {  2,   3,  66,  67},
    {  4,   5,  68,  69},
    {  6,   7,  70,  71},
    {  8,   9,  72,  73},
    { 10,  11,  74,  75},
    { 12,  13,  76,  77},
    { 14,  15,  78,  79},
    { 16,  17,  80,  81},
    { 18,  19,  82,  83},
    { 20,  21,  84,  85},
    { 22,  23,  86,  87},
    { 24,  25,  88,  89},
    { 26,  27,  90,  91},
    { 28,  29,  92,  93},
    { 30,  31,  94,  95},
    { 32,  33,  96,  97},
    { 34,  35,  98,  99},
    { 36,  37, 100, 101},
    { 38,  39, 102, 103},
    { 40,  41, 104, 105},
    { 42,  43, 106, 107},
    { 44,  45, 108, 109},
    { 46,  47, 110, 111},
    { 48,  49, 112, 113},
    { 50,  51, 114, 115},
    { 52,  53, 116, 117},
    { 54,  55, 118, 119},
    { 56,  57, 120, 121},
    { 58,  59, 122, 123},
    { 60,  61, 124, 125},
    { 62,  63, 126, 127},
};

// mma_m16n8k8 operand B cell = row * kCols + col.
constexpr int kM16OperandBCell[32][2] = {
    { 0,  8}, {16, 24}, {32, 40}, {48, 56},
    { 1,  9}, {17, 25}, {33, 41}, {49, 57},
    { 2, 10}, {18, 26}, {34, 42}, {50, 58},
    { 3, 11}, {19, 27}, {35, 43}, {51, 59},
    { 4, 12}, {20, 28}, {36, 44}, {52, 60},
    { 5, 13}, {21, 29}, {37, 45}, {53, 61},
    { 6, 14}, {22, 30}, {38, 46}, {54, 62},
    { 7, 15}, {23, 31}, {39, 47}, {55, 63},
};
#endif

constexpr int kWarpLanes = 32;

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
