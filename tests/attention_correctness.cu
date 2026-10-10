#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "test_utils.cuh"
#include "tk_sm7x/attention.cuh"

namespace {

constexpr int kWidth = 16;
constexpr int kCells = kWidth * kWidth;
constexpr double kSoftmaxBound = 512.0 * 0x1p-23;
constexpr double kEps = 0x1p-23;
constexpr double kGamma32 = (32.0 * kEps) / (1.0 - 32.0 * kEps);
constexpr float kGuardBefore = -12345.0f;
constexpr float kGuardAfter = 12345.0f;

struct Data {
    __half q[kCells];
    __half k[kCells];
    __half v[kCells];
    float guard_before;
    float out[kCells];
    float guard_after;
};

__global__ void seed_stale_launch_error() {}

std::size_t cell(int row, int col) {
    return static_cast<std::size_t>(row) * kWidth + col;
}

void fill_uniform(Data* data) {
    for (int row = 0; row < kWidth; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t index = cell(row, col);
            data->q[index] = __float2half_rn(0.0f);
            data->k[index] = __float2half_rn(static_cast<float>((7 * row + col) % 11 - 5) / 8.0f);
            data->v[index] = __float2half_rn(static_cast<float>((3 * row + 5 * col) % 13 - 6) / 8.0f);
        }
    }
}

void fill_shifted_one_hot(Data* data, bool identity_v) {
    for (int row = 0; row < kWidth; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t index = cell(row, col);
            data->q[index] = __float2half_rn(col == (row + 1) % kWidth ? 4.0f : 0.0f);
            data->k[index] = __float2half_rn(col == (row + 1) % kWidth ? 2.0f : 0.0f);
            data->v[index] = __float2half_rn(identity_v
                ? (row == col ? 1.0f : 0.0f)
                : static_cast<float>((3 * row + 5 * col) % 13 - 6) / 8.0f);
        }
    }
}

void fill_large_scores(Data* data) {
    for (int row = 0; row < kWidth; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t index = cell(row, col);
            data->q[index] = __float2half_rn(row == col ? 64.0f : 0.0f);
            data->k[index] = __float2half_rn(row == col ? 64.0f : 0.0f);
            data->v[index] = __float2half_rn(row == col ? 1.0f : 0.0f);
        }
    }
}

void fill_alias_input(Data* data) {
    for (int row = 0; row < kWidth; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t index = cell(row, col);
            data->q[index] = __float2half_rn(
                static_cast<float>((3 * row + 5 * col) % 13 - 6) / 8.0f);
        }
    }
}

void reset_output(Data* data) {
    data->guard_before = kGuardBefore;
    data->guard_after = kGuardAfter;
    for (float& value : data->out) value = std::numeric_limits<float>::quiet_NaN();
}

struct Reference {
    double probability[kCells];
    __half rounded_probability[kCells];
    double out[kCells];
};

Reference make_reference(const __half* q, const __half* k, const __half* v,
                         bool causal = false) {
    Reference reference{};
    for (int row = 0; row < kWidth; ++row) {
        double scores[kWidth];
        double maximum = -std::numeric_limits<double>::infinity();
        for (int token = 0; token < kWidth; ++token) {
            double dot = 0.0;
            for (int feature = 0; feature < kWidth; ++feature) {
                dot += static_cast<double>(__half2float(q[cell(row, feature)])) *
                       static_cast<double>(__half2float(k[cell(token, feature)]));
            }
            scores[token] = dot * 0.25;
            if ((!causal || token <= row) && scores[token] > maximum) {
                maximum = scores[token];
            }
        }
        double denominator = 0.0;
        for (int token = 0; token < kWidth; ++token) {
            if (causal && token > row) {
                scores[token] = 0.0;
            } else {
                scores[token] = std::exp(scores[token] - maximum);
                denominator += scores[token];
            }
        }
        for (int token = 0; token < kWidth; ++token) {
            const std::size_t index = cell(row, token);
            reference.probability[index] = scores[token] / denominator;
            reference.rounded_probability[index] =
                __double2half(reference.probability[index]);
        }
        for (int feature = 0; feature < kWidth; ++feature) {
            double output = 0.0;
            for (int token = 0; token < kWidth; ++token) {
                output += static_cast<double>(__half2float(
                              reference.rounded_probability[cell(row, token)])) *
                          static_cast<double>(__half2float(v[cell(token, feature)]));
            }
            reference.out[cell(row, feature)] = output;
        }
    }
    return reference;
}

double positive_half_value(std::uint16_t bits) {
    const int exponent = static_cast<int>((bits >> 10u) & 31u);
    const int fraction = static_cast<int>(bits & 1023u);
    return exponent == 0 ? std::ldexp(static_cast<double>(fraction), -24)
                         : std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
}

double midpoint_margin(double probability, __half rounded) {
    static_assert(sizeof(__half) == sizeof(std::uint16_t), "FP16 storage must be 16 bits");
    std::uint16_t bits = 0;
    std::memcpy(&bits, &rounded, sizeof(bits));
    const double value = static_cast<double>(__half2float(rounded));
    if (positive_half_value(bits) != value) return 0.0;
    const double lower = positive_half_value(static_cast<std::uint16_t>(bits - 1u));
    const double upper = positive_half_value(static_cast<std::uint16_t>(bits + 1u));
    return std::fmin(std::fabs(probability - (lower + value) / 2.0),
                     std::fabs(probability - (value + upper) / 2.0));
}

bool check_output(const char* name, const Data& data, const __half* v,
                  const Reference& reference, bool exact_uniform,
                  bool midpoint_favored, bool exact_large,
                  bool causal, bool identity_v) {
    if (data.guard_before != kGuardBefore || data.guard_after != kGuardAfter) {
        std::fprintf(stderr, "%s output guard changed\n", name);
        return false;
    }
    for (int row = 0; row < kWidth; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t index = cell(row, col);
            double value_scale = 0.0;
            for (int token = 0; token < kWidth; ++token) {
                value_scale += std::fabs(static_cast<double>(
                    __half2float(v[cell(token, col)])));
            }
            const double bound = (kSoftmaxBound + 0x1p-11 + kGamma32) * value_scale;
            const double got = static_cast<double>(data.out[index]);
            const double want = reference.out[index];
            const bool require_exact = exact_uniform || exact_large ||
                (midpoint_favored && row == col) || (causal && row == 0) ||
                (causal && identity_v && col > row);
            if (midpoint_favored && row == col &&
                midpoint_margin(reference.probability[index],
                                reference.rounded_probability[index]) <= kSoftmaxBound) {
                std::fprintf(stderr, "%s favored probability lacks midpoint margin at %d\n",
                             name, row);
                return false;
            }
            if (!std::isfinite(got) || std::fabs(got - want) > bound ||
                (require_exact && got != want)) {
                std::fprintf(stderr,
                             "%s [%d,%d] got=%.17g want=%.17g bound=%.17g exact=%d\n",
                             name, row, col, got, want, bound, require_exact ? 1 : 0);
                return false;
            }
        }
    }
    std::printf("%s: PASS\n", name);
    return true;
}

bool run_case(const char* name, Data* data, const __half* q, const __half* k,
              const __half* v, cudaStream_t stream, bool exact_uniform,
              bool midpoint_favored, bool exact_large, bool stale_error,
              bool causal = false, bool identity_v = false) {
    const Reference reference = make_reference(q, k, v, causal);
    reset_output(data);
    if (stale_error) {
        seed_stale_launch_error<<<0, 1>>>();
        if (cudaPeekAtLastError() == cudaSuccess) {
            std::fprintf(stderr, "%s could not seed stale CUDA error\n", name);
            return false;
        }
    }
    const cudaError_t launched = causal
        ? tk_sm7x::attention_forward_16x16_causal(q, k, v, data->out, stream)
        : tk_sm7x::attention_forward_16x16(q, k, v, data->out, stream);
    if (!tk_sm7x::test::cuda_ok(launched, "attention forward launch")) return false;
    if (!tk_sm7x::test::cuda_ok(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize")) return false;
    return check_output(name, *data, v, reference, exact_uniform,
                        midpoint_favored, exact_large, causal, identity_v);
}

bool check_null_arguments(Data* data, cudaStream_t stream, bool causal = false) {
    const __half* inputs[3] = {data->q, data->k, data->v};
    for (int missing = 0; missing < 4; ++missing) {
        const __half* q = missing == 0 ? nullptr : inputs[0];
        const __half* k = missing == 1 ? nullptr : inputs[1];
        const __half* v = missing == 2 ? nullptr : inputs[2];
        float* out = missing == 3 ? nullptr : data->out;
        const cudaError_t status = causal
            ? tk_sm7x::attention_forward_16x16_causal(q, k, v, out, stream)
            : tk_sm7x::attention_forward_16x16(q, k, v, out, stream);
        if (status != cudaErrorInvalidValue) {
            std::fprintf(stderr, "null argument %d returned %s\n",
                         missing, cudaGetErrorString(status));
            return false;
        }
    }
    std::printf("%snull arguments: PASS\n", causal ? "causal " : "");
    return true;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selected = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selected != EXIT_SUCCESS) return selected;

    Data* data = nullptr;
    if (!tk_sm7x::test::cuda_ok(cudaMallocManaged(&data, sizeof(Data)),
                                "cudaMallocManaged(data)")) return EXIT_FAILURE;
    cudaStream_t stream = nullptr;
    if (!tk_sm7x::test::cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                                "cudaStreamCreateWithFlags")) {
        static_cast<void>(cudaFree(data));
        return EXIT_FAILURE;
    }

    fill_uniform(data);
    bool ok = check_null_arguments(data, stream);
    ok = run_case("uniform scores", data, data->q, data->k, data->v,
                  stream, true, false, false, true) && ok;

    fill_shifted_one_hot(data, true);
    ok = run_case("shifted K, identity V", data, data->q, data->k, data->v,
                  stream, false, true, false, false) && ok;
    float unmasked_last_row[kWidth];
    for (int col = 0; col < kWidth; ++col) {
        unmasked_last_row[col] = data->out[cell(kWidth - 1, col)];
    }

    fill_shifted_one_hot(data, false);
    ok = run_case("shifted K, asymmetric V", data, data->q, data->k, data->v,
                  stream, false, false, false, false) && ok;

    fill_large_scores(data);
    ok = run_case("large finite scores", data, data->q, data->k, data->v,
                  stream, false, false, true, false) && ok;

    fill_alias_input(data);
    ok = run_case("aliased Q/K/V", data, data->q, data->q, data->q,
                  stream, false, false, false, false) && ok;

    fill_uniform(data);
    ok = check_null_arguments(data, stream, true) && ok;
    ok = run_case("causal uniform scores", data, data->q, data->k, data->v,
                  stream, false, false, false, true, true) && ok;

    fill_shifted_one_hot(data, true);
    const bool causal_identity = run_case("causal shifted K, identity V",
        data, data->q, data->k, data->v, stream,
        false, false, false, false, true, true);
    ok = causal_identity && ok;
    if (causal_identity) {
        for (int col = 0; col < kWidth; ++col) {
            const double got = static_cast<double>(data->out[cell(kWidth - 1, col)]);
            const double prior = static_cast<double>(unmasked_last_row[col]);
            const double bound = kSoftmaxBound + 0x1p-11 + kGamma32;
            if (!std::isfinite(prior) || std::fabs(got - prior) > bound) {
                std::fprintf(stderr, "causal last row %d got=%.17g noncausal=%.17g\n",
                             col, got, prior);
                ok = false;
            }
        }
    }

    fill_shifted_one_hot(data, false);
    ok = run_case("causal shifted K, asymmetric V", data,
                  data->q, data->k, data->v, stream,
                  false, false, false, false, true) && ok;

    fill_large_scores(data);
    ok = run_case("causal large finite scores", data,
                  data->q, data->k, data->v, stream,
                  false, false, true, false, true, true) && ok;

    fill_alias_input(data);
    ok = run_case("causal aliased Q/K/V", data,
                  data->q, data->q, data->q, stream,
                  false, false, false, false, true) && ok;

    ok = tk_sm7x::test::cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy") && ok;
    ok = tk_sm7x::test::cuda_ok(cudaFree(data), "cudaFree(data)") && ok;
    if (ok) std::printf("fixed attention: PASS ordinal=%d\n", ordinal);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
