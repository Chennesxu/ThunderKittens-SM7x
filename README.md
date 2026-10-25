# ThunderKittens-SM7x

ThunderKittens-SM7x is a small, independent CUDA C++ tile library inspired by
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens). It targets
NVIDIA Volta (SM70) and Turing (SM75), not the newer GPUs targeted by upstream.
It is **not** a drop-in port of upstream ThunderKittens or a general-purpose
FlashAttention implementation.

## Why this project?

[Upstream ThunderKittens](https://github.com/HazyResearch/ThunderKittens#hardware-requirements)
is built and tested mainly for newer GPUs, uses C++20 and CUDA 12.8+, and lists
test targets starting at SM80. An [upstream maintainer explained](https://github.com/HazyResearch/ThunderKittens/issues/21)
that pre-SM80 GPUs do not run its code as-is: SM75 needs changes around
asynchronous loads, FP16 support and MMA instructions, while SM70 needs further
layout and data-movement work. That is a support and maintenance choice, not a
claim that Volta or Turing cannot use Tensor Cores.

This project provides a deliberately narrower path for existing SM7x systems:
C++17/CUDA 11-compatible builds, synchronous staging, explicit WMMA and
architecture-specific PTX backends, and focused correctness and instruction
checks. Its value is legacy-hardware access and auditable tile code, not feature
parity with upstream.

## Support and limits

| Target | Status |
| --- | --- |
| SM75 (Turing) | Runtime-tested on TITAN RTX with CUDA 12.4. |
| SM70 (Volta) | Native compile/codegen-tested only; experimental until real Volta runtime validation. |
| CUDA 11.0 | Compile/codegen-tested in a pinned toolkit container; runtime not tested there. |

Matrix multiplication uses FP16 operands with FP32 accumulation; row operations
consume FP32 accumulators. The MMA tile is fixed at 16×16×16. This repository
does not support SM80+ facilities, other matrix input dtypes, arbitrary
Attention shapes or masks, batching, tails, or backward propagation.

## What works

| Operation | Entry point | Scope |
| --- | --- | --- |
| GEMM | [`gemm.cuh`](include/tk_sm7x/gemm.cuh) | Row-major FP16×FP16→FP32; positive 16-aligned M/N/K and valid leading dimensions. |
| Tile row sum/max | [`reduce.cuh`](include/tk_sm7x/reduce.cuh) | Reduce each row of a fixed 16×16 FP32 accumulator. |
| Tile row softmax | [`softmax.cuh`](include/tk_sm7x/softmax.cuh) | Full-row and fixed causal-prefix variants. |
| Attention forward | [`attention.cuh`](include/tk_sm7x/attention.cuh) | One 16-token × 16-feature tile, unmasked or causal; FP32 output. |

GEMM and Attention enqueue asynchronously on the supplied CUDA stream. Their
headers define pointer, aliasing, lifetime and error-return contracts; tile
operations also require the documented warp convergence and private shared
storage. Attention computes `softmax(QKᵀ / 4) V`, rounding probabilities to
FP16 nearest-even before the second MMA. It is a fixed-tile correctness slice,
not a variable-length sequence kernel.

WMMA is the default backend. Define `KITTENS_MMA_PTX` consistently across all
translation units using tile/backend headers to select native PTX; their
fragment representations must not cross mixed-backend translation units.
See the [row-statistics](examples/gemm_row_stats.cu) and
[GEMM-to-softmax](examples/gemm_row_softmax.cu) examples for tile composition.

## Build and verify

Use `nvcc` with C++17; CUDA 11.0 is the minimum tested toolkit. From the
repository root:

```bash
make all
make test-sm75 test-ptx-sm75
make test-attention-sm75 test-attention-ptx-sm75
```

`make all` builds native SM70 and SM75 targets and runs host, architecture and
codegen gates. SM75 runtime targets select an actual compute-capability 7.5
device; without one, Make prints **SKIP**, not PASS. SM70 builds do not establish
Volta runtime correctness.

Optional throughput measurements:

```bash
make bench-sm75 bench-ptx-sm75
make bench-attention-sm75 bench-attention-ptx-sm75
```

The Attention benchmark times repeated one-tile public calls after checking
every output. Its event time includes launch-feed effects and does not measure
isolated kernel latency or application throughput.

The [numerical validation notes](docs/numerical-validation.md) describe the
tested GEMM input domain and conditional error model. Fixture bounds and
backend comparisons are not universal NVIDIA accuracy guarantees.

## License

MIT; see [LICENSE](LICENSE). No upstream C++/CUDA implementation was copied
into the initial SM7x core.
