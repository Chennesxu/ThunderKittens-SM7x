#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(KITTENS_SM75)
#include "mma_layout_oracle.cuh"
#endif
#include "test_utils.cuh"
#include "tk_sm7x/ptx_backend.cuh"

namespace {

using backend_m8 =
    tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm70>;
#if defined(KITTENS_SM75)
using backend_m16 =
    tk_sm7x::detail::ptx_warp_mma_f16_f16_f32_16x16x16<tk_sm7x::arch::sm75>;
#endif
using tk_sm7x::test::cuda_ok;
#if defined(KITTENS_SM75)
using tk_sm7x::test::kM16FragmentCell;
using tk_sm7x::test::kM16OperandBCell;
#endif

constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kDepth = 16;
constexpr int kTileCells = kRows * kCols;
constexpr int kWarpLanes = 32;
constexpr int kMaxWarps = 4;
constexpr int kMaxInputLdm = 24;
constexpr int kMaxOutputLdm = 20;
constexpr uint32_t kNaNBits = 0x7fc12345u;

enum class operation {
    once,
    repeat_same,
    two_k_tiles,
    clear_reuse,
};

enum class staging {
    backend,
    packed_a,
    packed_b,
};

struct test_case {
    const char* name;
    int warps;
    int input_ldm;
    int output_ldm;
    operation op;
    staging stage;
    int fixture;
};

#if defined(KITTENS_SM75)
uint32_t pack_bits(__half low, __half high) {
    return static_cast<uint32_t>(static_cast<__half_raw>(low).x) |
           (static_cast<uint32_t>(static_cast<__half_raw>(high).x) << 16);
}
#endif

template <class Backend>
__global__ void run_case_kernel(
    const __half* a0, const __half* b0, const __half* a1, const __half* b1,
    const uint32_t* packed_a, const uint32_t* packed_b, float* output,
    int input_ldm, int output_ldm, operation op, staging stage) {
    __shared__ __align__(32) __half shared_a[kMaxWarps * kRows * kMaxInputLdm];
    __shared__ __align__(32) __half shared_b[kMaxWarps * kCols * kMaxInputLdm];
    __shared__ __align__(32) float shared_c[kMaxWarps * kRows * kMaxOutputLdm];

    const int lane = static_cast<int>(threadIdx.x) % kWarpLanes;
    const int warp = static_cast<int>(threadIdx.x) / kWarpLanes;
    __half* warp_a = shared_a + warp * kRows * kMaxInputLdm;
    __half* warp_b = shared_b + warp * kCols * kMaxInputLdm;
    float* warp_c = shared_c + warp * kRows * kMaxOutputLdm;
    const std::size_t input_base = static_cast<std::size_t>(warp) * kTileCells;
    const std::size_t output_base =
        static_cast<std::size_t>(warp) * kRows * output_ldm;

    for (int linear = lane; linear < kRows * input_ldm; linear += kWarpLanes) {
        const int row = linear / input_ldm;
        const int col = linear % input_ldm;
        if (col < kCols) {
            const std::size_t input_index =
                input_base + static_cast<std::size_t>(row) * kCols +
                static_cast<std::size_t>(col);
            warp_a[linear] = a0[input_index];
            warp_b[col * input_ldm + row] =
                b0[input_index];
        }
    }
    for (int linear = lane; linear < kRows * output_ldm; linear += kWarpLanes) {
        warp_c[linear] = __int_as_float(static_cast<int>(kNaNBits));
    }
    __syncwarp(0xffffffffu);

    typename Backend::fragment_a a_fragment;
    typename Backend::fragment_b b_fragment;
    if (stage == staging::packed_a) {
        const std::size_t base =
            (static_cast<std::size_t>(warp) * kWarpLanes +
             static_cast<std::size_t>(lane)) * 4;
        a_fragment.value[0][0] = packed_a[base];
        a_fragment.value[0][1] = packed_a[base + 1];
        a_fragment.value[1][0] = packed_a[base + 2];
        a_fragment.value[1][1] = packed_a[base + 3];
    } else {
        Backend::load_a(a_fragment, warp_a, input_ldm);
    }
    if (stage == staging::packed_b) {
        const std::size_t base =
            (static_cast<std::size_t>(warp) * kWarpLanes +
             static_cast<std::size_t>(lane)) * 4;
        b_fragment.value[0][0] = packed_b[base];
        b_fragment.value[0][1] = packed_b[base + 1];
        b_fragment.value[1][0] = packed_b[base + 2];
        b_fragment.value[1][1] = packed_b[base + 3];
    } else {
        Backend::load_b(b_fragment, warp_b, input_ldm);
    }

    typename Backend::accumulator accumulator;
    Backend::clear(accumulator);
    Backend::mma(accumulator, a_fragment, b_fragment);
    if (op == operation::repeat_same) {
        Backend::mma(accumulator, a_fragment, b_fragment);
    } else if (op == operation::two_k_tiles || op == operation::clear_reuse) {
        __syncwarp(0xffffffffu);
        for (int linear = lane; linear < kTileCells; linear += kWarpLanes) {
            const int row = linear / kCols;
            const int col = linear % kCols;
            const std::size_t input_index =
                input_base + static_cast<std::size_t>(linear);
            warp_a[row * input_ldm + col] = a1[input_index];
            warp_b[col * input_ldm + row] = b1[input_index];
        }
        __syncwarp(0xffffffffu);
        Backend::load_a(a_fragment, warp_a, input_ldm);
        Backend::load_b(b_fragment, warp_b, input_ldm);
        if (op == operation::clear_reuse) {
            Backend::clear(accumulator);
        }
        Backend::mma(accumulator, a_fragment, b_fragment);
    }

    Backend::store(warp_c, accumulator, output_ldm);
    __syncwarp(0xffffffffu);
    for (int linear = lane; linear < kRows * output_ldm; linear += kWarpLanes) {
        const std::size_t output_index =
            output_base + static_cast<std::size_t>(linear);
        output[output_index] = warp_c[linear];
    }
}

int fixture_value(int fixture, int matrix, int warp, int row, int col) {
    if (fixture == 0 && matrix == 0) {
        return row == col ? 1 : 0;
    }
    if (fixture == 0) {
        return (row * 5 + col * 3 + warp * 2) % 7 - 3;
    }
    const int salt = fixture * 3 + matrix * 5 + warp * 7;
    const int row_scale = 2 + ((fixture + matrix) % 3);
    const int col_scale = 3 + ((2 * fixture + matrix) % 4);
    return (row * row_scale + col * col_scale + salt) % 9 - 4;
}

void fill_matrix(std::vector<__half>& values, int warps, int fixture, int matrix) {
    for (int warp = 0; warp < warps; ++warp) {
        const std::size_t base = static_cast<std::size_t>(warp) * kTileCells;
        for (int row = 0; row < kRows; ++row) {
            for (int col = 0; col < kCols; ++col) {
                values[base + row * kCols + col] = __float2half(
                    static_cast<float>(fixture_value(fixture, matrix, warp, row, col)));
            }
        }
    }
}

#if defined(KITTENS_SM75)
void pack_a_from_oracle(const std::vector<__half>& a, int warps,
                        std::vector<uint32_t>& packed) {
    for (int warp = 0; warp < warps; ++warp) {
        const std::size_t tile_base = static_cast<std::size_t>(warp) * kTileCells;
        for (int lane = 0; lane < kWarpLanes; ++lane) {
            const std::size_t packed_base =
                (static_cast<std::size_t>(warp) * kWarpLanes + lane) * 4;
            for (int kh = 0; kh < 2; ++kh) {
                for (int reg = 0; reg < 2; ++reg) {
                    const int low_cell = kM16FragmentCell[lane][2 * reg];
                    const int high_cell = kM16FragmentCell[lane][2 * reg + 1];
                    const int low_row = low_cell / 8;
                    const int high_row = high_cell / 8;
                    const int low_col = kh * 8 + low_cell % 8;
                    const int high_col = kh * 8 + high_cell % 8;
                    packed[packed_base + kh * 2 + reg] = pack_bits(
                        a[tile_base + low_row * kCols + low_col],
                        a[tile_base + high_row * kCols + high_col]);
                }
            }
        }
    }
}

void pack_b_from_oracle(const std::vector<__half>& b, int warps,
                        std::vector<uint32_t>& packed) {
    for (int warp = 0; warp < warps; ++warp) {
        const std::size_t tile_base = static_cast<std::size_t>(warp) * kTileCells;
        for (int lane = 0; lane < kWarpLanes; ++lane) {
            const std::size_t packed_base =
                (static_cast<std::size_t>(warp) * kWarpLanes + lane) * 4;
            for (int kh = 0; kh < 2; ++kh) {
                for (int nh = 0; nh < 2; ++nh) {
                    const int low_cell = kM16OperandBCell[lane][0];
                    const int high_cell = kM16OperandBCell[lane][1];
                    const int low_row = kh * 8 + low_cell / 8;
                    const int high_row = kh * 8 + high_cell / 8;
                    const int low_col = nh * 8 + low_cell % 8;
                    const int high_col = nh * 8 + high_cell % 8;
                    packed[packed_base + kh * 2 + nh] = pack_bits(
                        b[tile_base + low_row * kCols + low_col],
                        b[tile_base + high_row * kCols + high_col]);
                }
            }
        }
    }
}
#endif

void accumulate_reference(const std::vector<__half>& a, const std::vector<__half>& b,
                          int warps, float scale, std::vector<float>& reference) {
    for (int warp = 0; warp < warps; ++warp) {
        const std::size_t base = static_cast<std::size_t>(warp) * kTileCells;
        for (int row = 0; row < kRows; ++row) {
            for (int col = 0; col < kCols; ++col) {
                float sum = 0.0f;
                for (int kk = 0; kk < kDepth; ++kk) {
                    sum += __half2float(a[base + row * kCols + kk]) *
                           __half2float(b[base + kk * kCols + col]);
                }
                reference[base + row * kCols + col] += scale * sum;
            }
        }
    }
}

template <class Backend>
bool run_case(const test_case& spec, int ordinal, const char* backend_name) {
    const std::size_t input_cells =
        static_cast<std::size_t>(spec.warps) * kTileCells;
    const std::size_t output_cells =
        static_cast<std::size_t>(spec.warps) * kRows * spec.output_ldm;
    const std::size_t packed_words =
        static_cast<std::size_t>(spec.warps) * kWarpLanes * 4;
    std::vector<__half> a0(input_cells);
    std::vector<__half> b0(input_cells);
    std::vector<__half> a1(input_cells);
    std::vector<__half> b1(input_cells);
    std::vector<uint32_t> packed_a(packed_words);
    std::vector<uint32_t> packed_b(packed_words);
    std::vector<float> actual(output_cells);
    std::vector<float> reference(input_cells, 0.0f);

    fill_matrix(a0, spec.warps, spec.fixture, 0);
    fill_matrix(b0, spec.warps, spec.fixture, 1);
    fill_matrix(a1, spec.warps, spec.fixture + 5, 2);
    fill_matrix(b1, spec.warps, spec.fixture + 5, 3);
#if defined(KITTENS_SM75)
    if (spec.stage == staging::packed_a) {
        pack_a_from_oracle(a0, spec.warps, packed_a);
    }
    if (spec.stage == staging::packed_b) {
        pack_b_from_oracle(b0, spec.warps, packed_b);
    }
#endif
    if (spec.op == operation::clear_reuse) {
        accumulate_reference(a1, b1, spec.warps, 1.0f, reference);
    } else {
        const float scale = spec.op == operation::repeat_same ? 2.0f : 1.0f;
        accumulate_reference(a0, b0, spec.warps, scale, reference);
        if (spec.op == operation::two_k_tiles) {
            accumulate_reference(a1, b1, spec.warps, 1.0f, reference);
        }
    }

    __half* device_a0 = nullptr;
    __half* device_b0 = nullptr;
    __half* device_a1 = nullptr;
    __half* device_b1 = nullptr;
    uint32_t* device_packed_a = nullptr;
    uint32_t* device_packed_b = nullptr;
    float* device_output = nullptr;
    bool ok = cuda_ok(cudaMalloc(&device_a0, input_cells * sizeof(__half)),
                      "cudaMalloc(A0)") &&
              cuda_ok(cudaMalloc(&device_b0, input_cells * sizeof(__half)),
                      "cudaMalloc(B0)") &&
              cuda_ok(cudaMalloc(&device_a1, input_cells * sizeof(__half)),
                      "cudaMalloc(A1)") &&
              cuda_ok(cudaMalloc(&device_b1, input_cells * sizeof(__half)),
                      "cudaMalloc(B1)") &&
              cuda_ok(cudaMalloc(&device_packed_a, packed_words * sizeof(uint32_t)),
                      "cudaMalloc(packed A)") &&
              cuda_ok(cudaMalloc(&device_packed_b, packed_words * sizeof(uint32_t)),
                      "cudaMalloc(packed B)") &&
              cuda_ok(cudaMalloc(&device_output, output_cells * sizeof(float)),
                      "cudaMalloc(output)");
    if (ok) {
        ok = cuda_ok(cudaMemcpy(device_a0, a0.data(), input_cells * sizeof(__half),
                                cudaMemcpyHostToDevice), "cudaMemcpy(A0)") &&
             cuda_ok(cudaMemcpy(device_b0, b0.data(), input_cells * sizeof(__half),
                                cudaMemcpyHostToDevice), "cudaMemcpy(B0)") &&
             cuda_ok(cudaMemcpy(device_a1, a1.data(), input_cells * sizeof(__half),
                                cudaMemcpyHostToDevice), "cudaMemcpy(A1)") &&
             cuda_ok(cudaMemcpy(device_b1, b1.data(), input_cells * sizeof(__half),
                                cudaMemcpyHostToDevice), "cudaMemcpy(B1)") &&
             cuda_ok(cudaMemcpy(device_packed_a, packed_a.data(),
                                packed_words * sizeof(uint32_t), cudaMemcpyHostToDevice),
                     "cudaMemcpy(packed A)") &&
             cuda_ok(cudaMemcpy(device_packed_b, packed_b.data(),
                                packed_words * sizeof(uint32_t), cudaMemcpyHostToDevice),
                     "cudaMemcpy(packed B)");
    }
    if (ok) {
        static_cast<void>(cudaGetLastError());
        run_case_kernel<Backend><<<1, spec.warps * kWarpLanes>>>(
            device_a0, device_b0, device_a1, device_b1, device_packed_a,
            device_packed_b, device_output, spec.input_ldm, spec.output_ldm,
            spec.op, spec.stage);
        ok = cuda_ok(cudaGetLastError(), "PTX MMA test launch") &&
             cuda_ok(cudaDeviceSynchronize(), "PTX MMA test synchronize") &&
             cuda_ok(cudaMemcpy(actual.data(), device_output,
                                output_cells * sizeof(float), cudaMemcpyDeviceToHost),
                     "cudaMemcpy(output)");
    }
    if (ok) {
        for (int warp = 0; warp < spec.warps && ok; ++warp) {
            const std::size_t result_base =
                static_cast<std::size_t>(warp) * kRows * spec.output_ldm;
            const std::size_t reference_base =
                static_cast<std::size_t>(warp) * kTileCells;
            for (int row = 0; row < kRows && ok; ++row) {
                for (int col = 0; col < spec.output_ldm; ++col) {
                    const float observed = actual[result_base + row * spec.output_ldm + col];
                    if (col < kCols) {
                        const float expected = reference[reference_base + row * kCols + col];
                        if (!std::isfinite(observed) || observed != expected) {
                            std::fprintf(stderr,
                                         "%s %s mismatch warp=%d row=%d col=%d "
                                         "got=%g want=%g\n",
                                         backend_name, spec.name, warp, row, col,
                                         observed, expected);
                            ok = false;
                            break;
                        }
                    } else if (!std::isnan(observed)) {
                        std::fprintf(stderr,
                                     "%s %s padding modified warp=%d row=%d col=%d "
                                     "got=%g\n",
                                     backend_name, spec.name, warp, row, col, observed);
                        ok = false;
                        break;
                    }
                }
            }
        }
    }

    bool released = cuda_ok(cudaFree(device_a0), "cudaFree(A0)");
    released = cuda_ok(cudaFree(device_b0), "cudaFree(B0)") && released;
    released = cuda_ok(cudaFree(device_a1), "cudaFree(A1)") && released;
    released = cuda_ok(cudaFree(device_b1), "cudaFree(B1)") && released;
    released = cuda_ok(cudaFree(device_packed_a), "cudaFree(packed A)") && released;
    released = cuda_ok(cudaFree(device_packed_b), "cudaFree(packed B)") && released;
    released = cuda_ok(cudaFree(device_output), "cudaFree(output)") && released;
    if (ok && released) {
        std::printf("%s %s: PASS ordinal=%d\n", backend_name, spec.name, ordinal);
    }
    return ok && released;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selection = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selection != EXIT_SUCCESS) {
        return selection;
    }

    const test_case common_cases[] = {
        {"identity", 1, 16, 16, operation::once, staging::backend, 0},
        {"fingerprint", 1, 16, 16, operation::once, staging::backend, 1},
        {"repeat-same", 1, 16, 16, operation::repeat_same, staging::backend, 2},
        {"two-k-tiles", 1, 16, 16, operation::two_k_tiles, staging::backend, 3},
        {"clear-reuse", 1, 16, 16, operation::clear_reuse, staging::backend, 4},
        {"padded-ld", 1, 24, 20, operation::once, staging::backend, 5},
        {"multiwarp", 4, 16, 16, operation::once, staging::backend, 6},
    };
    for (const test_case& spec : common_cases) {
        if (!run_case<backend_m8>(spec, ordinal, "ptx m8")) {
            return EXIT_FAILURE;
        }
    }
#if defined(KITTENS_SM75)
    for (const test_case& spec : common_cases) {
        if (!run_case<backend_m16>(spec, ordinal, "ptx m16")) {
            return EXIT_FAILURE;
        }
    }
    const test_case mixed_cases[] = {
        {"scalar-a-hardware-b", 1, 24, 20, operation::once, staging::packed_a, 7},
        {"hardware-a-scalar-b", 4, 24, 20, operation::once, staging::packed_b, 8},
    };
    for (const test_case& spec : mixed_cases) {
        if (!run_case<backend_m16>(spec, ordinal, "ptx m16")) {
            return EXIT_FAILURE;
        }
    }
#endif
    std::printf("ptx logical backends: PASS ordinal=%d\n", ordinal);
    return EXIT_SUCCESS;
}
