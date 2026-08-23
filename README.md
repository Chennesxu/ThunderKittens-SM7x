# ThunderKittens-SM7x

**ThunderKittens-SM7x** is an independent, community-maintained derivative of [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) focused on NVIDIA Volta (SM70) and Turing (SM75) GPUs.

Upstream ThunderKittens now targets SM80 and newer architectures, with most active development focused on Hopper and Blackwell. Its current codebase does not support pre-SM80 GPUs. Community interest in SM70 and SM75 support is tracked in [ThunderKittens issue #21](https://github.com/HazyResearch/ThunderKittens/issues/21).

This repository aims to maintain a small, architecture-specific implementation for SM70 and SM75 instead of backporting features that require newer hardware.

## Goals

1. Provide reusable tile primitives for Volta and Turing.
2. Support FP16 Tensor Core operations with FP32 accumulation.
3. Replace SM80+ memory and synchronization paths with SM7x-compatible implementations.
4. Keep the programming model close to ThunderKittens where practical.

## Scope

- NVIDIA SM70 and SM75 only
- CUDA 11.0+
- C++17
- SM75 is the primary tested target; SM70 will remain experimental until it is validated on real Volta hardware

## Status

Early development. There is no usable release yet.

## Upstream and License

The initial port is based on [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) at commit [`0230013a`](https://github.com/HazyResearch/ThunderKittens/commit/0230013a72b51338a137b50f69538ec69d4d4675).

ThunderKittens is Copyright (c) 2024–2026 HazyResearch and is distributed under the [MIT License](https://github.com/HazyResearch/ThunderKittens/blob/main/LICENSE). Original copyright and license notices will be retained when code is imported.

This is an independent community project and is not affiliated with or endorsed by HazyResearch.
