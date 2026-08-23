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

Early development. There is no usable release yet.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
