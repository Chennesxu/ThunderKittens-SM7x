# ThunderKittens-SM7x

**ThunderKittens-SM7x** is an independent project based on [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), focused on NVIDIA Volta (SM70) and Turing (SM75) GPUs.

Upstream ThunderKittens now targets SM80 and newer architectures, with most active development focused on Hopper and Blackwell. Its current codebase does not support pre-SM80 GPUs. Community interest in SM70 and SM75 support is tracked in [ThunderKittens issue #21](https://github.com/HazyResearch/ThunderKittens/issues/21).

This repository aims to maintain a small, architecture-specific implementation for SM70 and SM75 instead of backporting features that require newer hardware.

## Goals

1. Bring the ThunderKittens tile programming model to Volta and Turing.
2. Provide architecture-specific backends built around each target's native Tensor Core instructions and memory hierarchy.
3. Implement pipelines and synchronization paths that do not depend on `cp.async`, TMA, or other SM80+ features.
4. Keep a common high-level API across SM70 and SM75 while making hardware capability differences explicit.
5. Provide tested building blocks and examples for custom kernels, beginning with GEMM and later expanding to attention workloads.
6. Maintain a stable CUDA 11+ codebase for legacy GPU users as upstream development continues toward newer architectures.
7. Improve hardware coverage through community testing, especially for V100 and TITAN V systems not available to the maintainer.

## Scope

- NVIDIA SM70 and SM75 only
- CUDA 11.0+
- C++17
- SM75 is the primary tested target; SM70 will remain experimental until it is validated on real Volta hardware

## Status

The first WMMA reference baseline builds native SM70 and SM75 targets and provides row-major FP16 × FP16 → FP32 GEMM for positive 16-aligned dimensions and non-compact leading dimensions.

The API enqueues asynchronously on the supplied CUDA stream. A, B, and C must be device-accessible, alive through stream completion, and non-overlapping; before completion, A/B cannot be modified and C cannot be read or written except by accesses ordered with the GEMM on that stream.

WMMA remains the default backend. Defining `KITTENS_MMA_PTX` opts a build into
the native inline-PTX backend for its selected SM70 or SM75 target. Apply that
definition consistently to every translation unit using the tile or backend
headers: WMMA and PTX fragments have different representations and must not
cross a translation-unit boundary compiled with different backend selections.

The repository provides separate builds for the opt-in path:

```text
make build-ptx-sm70 build-ptx-sm75
make test-ptx-sm75
make sanitize-ptx-sm75
make build-bench-ptx-sm75
make bench-ptx-sm75
```

The SM70 PTX target is compile- and codegen-checked only; opting in does not
change its experimental status or expand the supported architecture set.

The differential GEMM driver builds the same six exact-domain cases with the
WMMA, PTX m8, and (on SM75) PTX m16 backends. On SM75 it also compares the
selected production GEMM path on the same allocations:

```text
make build-differential-sm70 build-differential-sm75
make build-differential-ptx-sm70 build-differential-ptx-sm75
make test-differential-sm75 test-differential-ptx-sm75
make sanitize-differential-sm75 sanitize-differential-ptx-sm75
```

Its inputs are FP16 values q/8 for integer |q| <= 8 and K <= 1024. Products
are exact multiples of 1/64 and the sum of product magnitudes is at most K, so
the bounded partial sums remain exactly representable in FP32; the independent
CPU reference uses double accumulation. This justifies finite zero-tolerance
comparisons for these fixtures, including direct backend pairs and row-padding
sentinels. It is not a general floating-point accuracy claim: arbitrary FP16
inputs and their error budget remain outside this exact-domain check.

- SM75 correctness and Compute Sanitizer checks pass on the identified Turing device.
- Compile and codegen checks pass with CUDA 11.0.3 and the local CUDA 12.4 toolkit.
- SM70 is compile-tested and remains experimental until it is run on Volta hardware.
- No performance claims are made for this correctness baseline.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
