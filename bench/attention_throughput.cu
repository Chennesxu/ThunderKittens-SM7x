#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "test_utils.cuh"
#include "tk_sm7x/attention.cuh"

namespace {

constexpr int kWidth = 16;
constexpr int kCells = kWidth * kWidth;
constexpr double kSoftmaxBound = 512.0 * 0x1p-23;
constexpr double kEps = 0x1p-23;
constexpr double kGamma32 = (32.0 * kEps) / (1.0 - 32.0 * kEps);
constexpr int kWarmupCalls = 32;
constexpr int kTimedCalls = 2048;
constexpr int kRounds = 4;

std::size_t cell(int row, int col) {
    return static_cast<std::size_t>(row) * kWidth + col;
}

struct Inputs {
    std::array<__half, kCells> q{};
    std::array<__half, kCells> k{};
    std::array<__half, kCells> v{};
};

Inputs make_inputs() {
    Inputs inputs;
    for (int row = 0; row < kWidth; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            inputs.q[cell(row, col)] = __float2half_rn(
                col == (row + 1) % kWidth ? 4.0f : 0.0f);
            inputs.k[cell(row, col)] = __float2half_rn(
                col == (row + 3) % kWidth ? 2.0f : 0.0f);
            inputs.v[cell(row, col)] = __float2half_rn(
                static_cast<float>((3 * row + 5 * col) % 13 - 6) / 8.0f);
        }
    }
    return inputs;
}

std::array<double, kCells> reference_output(const Inputs& inputs, bool causal) {
    std::array<double, kCells> output{};
    for (int row = 0; row < kWidth; ++row) {
        double scores[kWidth];
        double maximum = -std::numeric_limits<double>::infinity();
        for (int token = 0; token < kWidth; ++token) {
            double dot = 0.0;
            for (int feature = 0; feature < kWidth; ++feature) {
                dot += static_cast<double>(__half2float(inputs.q[cell(row, feature)])) *
                       static_cast<double>(__half2float(inputs.k[cell(token, feature)]));
            }
            scores[token] = dot * 0.25;
            if ((!causal || token <= row) && scores[token] > maximum) {
                maximum = scores[token];
            }
        }
        double denominator = 0.0;
        for (int token = 0; token < kWidth; ++token) {
            if (!causal || token <= row) {
                scores[token] = std::exp(scores[token] - maximum);
                denominator += scores[token];
            } else {
                scores[token] = 0.0;
            }
        }
        for (int feature = 0; feature < kWidth; ++feature) {
            double sum = 0.0;
            for (int token = 0; token < kWidth; ++token) {
                const __half probability = __double2half(scores[token] / denominator);
                sum += static_cast<double>(__half2float(probability)) *
                       static_cast<double>(__half2float(inputs.v[cell(token, feature)]));
            }
            output[cell(row, feature)] = sum;
        }
    }
    return output;
}

bool check_output(const char* name, const Inputs& inputs,
                  const std::array<float, kCells>& got, bool causal) {
    const auto reference = reference_output(inputs, causal);
    for (int row = 0; row < kWidth; ++row) {
        for (int feature = 0; feature < kWidth; ++feature) {
            double scale = 0.0;
            for (int token = 0; token < kWidth; ++token) {
                scale += std::fabs(static_cast<double>(
                    __half2float(inputs.v[cell(token, feature)])));
            }
            const std::size_t index = cell(row, feature);
            const double bound = (kSoftmaxBound + 0x1p-11 + kGamma32) * scale;
            const double value = static_cast<double>(got[index]);
            if (!std::isfinite(value) || std::fabs(value - reference[index]) > bound) {
                std::fprintf(stderr, "%s output [%d,%d] got=%.17g want=%.17g bound=%.17g\n",
                             name, row, feature, value, reference[index], bound);
                return false;
            }
        }
    }
    std::printf("%s output: PASS 256/256\n", name);
    return true;
}

bool release(__half* q, __half* k, __half* v, float* out) {
    bool ok = true;
    if (q != nullptr) ok = tk_sm7x::test::cuda_ok(cudaFree(q), "cudaFree(q)") && ok;
    if (k != nullptr) ok = tk_sm7x::test::cuda_ok(cudaFree(k), "cudaFree(k)") && ok;
    if (v != nullptr) ok = tk_sm7x::test::cuda_ok(cudaFree(v), "cudaFree(v)") && ok;
    if (out != nullptr) ok = tk_sm7x::test::cuda_ok(cudaFree(out), "cudaFree(out)") && ok;
    return ok;
}

cudaError_t launch(bool causal, const __half* q, const __half* k,
                   const __half* v, float* out) {
    return causal ? tk_sm7x::attention_forward_16x16_causal(q, k, v, out)
                  : tk_sm7x::attention_forward_16x16(q, k, v, out);
}

bool measure(bool causal, const __half* q, const __half* k,
             const __half* v, float* out, double* microseconds_per_call) {
    for (int call = 0; call < kWarmupCalls; ++call) {
        if (!tk_sm7x::test::cuda_ok(launch(causal, q, k, v, out), "warmup launch")) {
            return false;
        }
    }
    if (!tk_sm7x::test::cuda_ok(cudaDeviceSynchronize(), "warmup synchronize")) {
        return false;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!tk_sm7x::test::cuda_ok(cudaEventCreate(&start), "cudaEventCreate(start)")) {
        return false;
    }
    if (!tk_sm7x::test::cuda_ok(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        static_cast<void>(tk_sm7x::test::cuda_ok(cudaEventDestroy(start),
                                                 "cudaEventDestroy(start)"));
        return false;
    }

    bool ok = tk_sm7x::test::cuda_ok(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int call = 0; ok && call < kTimedCalls; ++call) {
        ok = tk_sm7x::test::cuda_ok(launch(causal, q, k, v, out), "timed launch");
    }
    if (ok) ok = tk_sm7x::test::cuda_ok(cudaEventRecord(stop), "cudaEventRecord(stop)");
    if (ok) ok = tk_sm7x::test::cuda_ok(cudaEventSynchronize(stop),
                                        "cudaEventSynchronize(stop)");
    float milliseconds = 0.0f;
    if (ok) ok = tk_sm7x::test::cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop),
                                        "cudaEventElapsedTime");
    ok = tk_sm7x::test::cuda_ok(cudaEventDestroy(start), "cudaEventDestroy(start)") && ok;
    ok = tk_sm7x::test::cuda_ok(cudaEventDestroy(stop), "cudaEventDestroy(stop)") && ok;
    if (!ok) return false;
    *microseconds_per_call = static_cast<double>(milliseconds) * 1000.0 / kTimedCalls;
    return std::isfinite(*microseconds_per_call) && *microseconds_per_call > 0.0;
}

}  // namespace

int main() {
    int ordinal = -1;
    const int selected = tk_sm7x::test::select_sm75_device(&ordinal);
    if (selected != EXIT_SUCCESS) return selected;

    const Inputs inputs = make_inputs();
    __half* q = nullptr;
    __half* k = nullptr;
    __half* v = nullptr;
    float* out = nullptr;
    bool ok = tk_sm7x::test::cuda_ok(cudaMalloc(&q, kCells * sizeof(__half)), "cudaMalloc(q)") &&
              tk_sm7x::test::cuda_ok(cudaMalloc(&k, kCells * sizeof(__half)), "cudaMalloc(k)") &&
              tk_sm7x::test::cuda_ok(cudaMalloc(&v, kCells * sizeof(__half)), "cudaMalloc(v)") &&
              tk_sm7x::test::cuda_ok(cudaMalloc(&out, kCells * sizeof(float)), "cudaMalloc(out)");
    if (ok) {
        ok = tk_sm7x::test::cuda_ok(cudaMemcpy(q, inputs.q.data(), kCells * sizeof(__half),
                                               cudaMemcpyHostToDevice), "cudaMemcpy(q)") &&
             tk_sm7x::test::cuda_ok(cudaMemcpy(k, inputs.k.data(), kCells * sizeof(__half),
                                               cudaMemcpyHostToDevice), "cudaMemcpy(k)") &&
             tk_sm7x::test::cuda_ok(cudaMemcpy(v, inputs.v.data(), kCells * sizeof(__half),
                                               cudaMemcpyHostToDevice), "cudaMemcpy(v)");
    }
    for (bool causal : {false, true}) {
        if (!ok) break;
        const char* name = causal ? "causal" : "unmasked";
        ok = tk_sm7x::test::cuda_ok(cudaMemset(out, 0xff, kCells * sizeof(float)),
                                    "cudaMemset(out)");
        if (!ok) break;
        ok = tk_sm7x::test::cuda_ok(launch(causal, q, k, v, out), "attention launch") &&
             tk_sm7x::test::cuda_ok(cudaDeviceSynchronize(), "attention synchronize");
        if (!ok) break;
        std::array<float, kCells> host_output{};
        ok = tk_sm7x::test::cuda_ok(cudaMemcpy(host_output.data(), out,
                                               kCells * sizeof(float), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy(out)") && ok;
        if (ok) ok = check_output(name, inputs, host_output, causal);
    }
    std::array<std::array<double, kRounds>, 2> samples{};
    for (int round = 0; ok && round < kRounds; ++round) {
        for (int order = 0; order < 2; ++order) {
            const int mode = round % 2 == 0 ? order : 1 - order;
            ok = measure(mode == 1, q, k, v, out, &samples[mode][round]);
            if (!ok) {
                std::fprintf(stderr, "%s measurement unavailable\n",
                             mode == 1 ? "causal" : "unmasked");
                break;
            }
        }
    }
    if (ok) {
        double medians[2] = {};
        for (int mode = 0; mode < 2; ++mode) {
            auto sorted = samples[mode];
            std::sort(sorted.begin(), sorted.end());
            medians[mode] = (sorted[1] + sorted[2]) / 2.0;
            std::printf("%s median=%.4f range=[%.4f,%.4f] us/public-call "
                        "throughput=%.0f tiles/s\n",
                        mode == 1 ? "causal" : "unmasked", medians[mode],
                        sorted.front(), sorted.back(), 1.0e6 / medians[mode]);
        }
        std::printf("causal/unmasked median time ratio=%.4f\n",
                    medians[1] / medians[0]);
        std::printf("attention benchmark: %d launches/round x %d rounds, "
                    "one 16x16 tile per launch, ordinal=%d\n",
                    kTimedCalls, kRounds, ordinal);
    }
    return release(q, k, v, out) && ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
