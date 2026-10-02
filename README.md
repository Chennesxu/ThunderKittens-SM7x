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

### Tile row statistics

The public `row_sum` and `row_max` operations reduce the 16 columns of an
`rt_c` accumulator, writing 16 contiguous FP32 values indexed by logical row.
The same signatures work in the default WMMA and opt-in PTX builds. A one-warp
16×16×16 GEMM using them is in [examples/gemm_row_stats.cu](examples/gemm_row_stats.cu):

```cpp
tk_sm7x::rt_a a_tile;
tk_sm7x::rt_b b_tile;
tk_sm7x::rt_c c_tile;
tk_sm7x::load(a_tile, shared_a, 0);
tk_sm7x::load(b_tile, shared_b, 0);
tk_sm7x::zero(c_tile);
tk_sm7x::mma(c_tile, a_tile, b_tile);
tk_sm7x::store(shared_c, c_tile);
__syncwarp();
tk_sm7x::store_block<32>(tk_sm7x::gl<float>{c, 16, 16, 16}, shared_c, 0, 0);
tk_sm7x::row_sum(sums, c_tile, scratch);
tk_sm7x::row_max(maxima, c_tile, scratch);
```

Here `shared_a` is a row-major FP16 tile, `shared_b` is a column-major FP16
tile, and `shared_c` and `scratch` are separate row-major FP32 tiles. The caller
allocates `scratch` explicitly in shared memory: one 16×16 tile is 1024 bytes
per warp. WMMA uses it to store and reduce the accumulator; the PTX reduction
does not use it, but takes the same argument. `sums` and `maxima` each point to
16 writable FP32 values in shared or global memory. They must not overlap
`scratch`, and concurrent warps need private scratch and disjoint destinations.
The 32 lanes of each warp must call together in a one-dimensional CTA with a
block size divisible by 32. Each call completes its writes and permits the
calling warp to consume the result or reuse scratch on return. Cross-warp and
cross-CTA consumers need their own synchronization. Inputs and sum
intermediates must be finite; the reduction uses a balanced column-order tree,
with maximum ties choosing the right operand.

```text
make build-example-row-stats-sm70 build-example-row-stats-sm75
make build-example-row-stats-ptx-sm70 build-example-row-stats-ptx-sm75
make example-row-stats-sm75 example-row-stats-ptx-sm75
```

The example writes all 256 FP32 C values and both 16-value statistics arrays,
then checks them against an exact q/8 input reference. SM70 example builds are
compile-only; runtime validation remains limited to SM75 hardware. An exit 77
from an SM75 binary is reported by Make as SKIP.

### Tile row softmax

Include `tk_sm7x/softmax.cuh` to normalize each of the 16 rows of an `rt_c`:

```cpp
__shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> probabilities;
__shared__ tk_sm7x::st<float, 16, 16, tk_sm7x::row_major> scratch;
tk_sm7x::row_softmax(probabilities, c_tile, scratch);
```

The accumulator must contain finite logits and is preserved. The output and
scratch must be distinct, aligned shared tiles private to the calling warp.
All 32 lanes call convergently in a one-dimensional CTA whose block size is
divisible by 32. The output is row-major FP32, computed with maximum subtraction,
CUDA `expf`, normal FP32 division and balanced column-order reductions. On return,
the warp can read the output and reuse scratch; consumers in other warps or CTAs
require caller synchronization. WMMA stores the opaque accumulator to scratch;
the opt-in PTX path uses its backend register layout and warp shuffles.

The [public correctness test](tests/tile_softmax.cu) uses FP16 scores times an
identity matrix, verifies the exact accumulator logits, then compares with an
independent binary64 softmax. Its predeclared absolute bound is `512 * 2^-23`
for each probability and the row sum, only for these identity-MMA fixtures:
position-distinguishable q/8 logits, zero rows and offsets near ±1000. This is
neither a universal accuracy guarantee nor the GEMM model budget.

```text
make build-tile-softmax-sm70 build-tile-softmax-ptx-sm70
make test-tile-softmax-sm75 test-tile-softmax-ptx-sm75
make sanitize-tile-softmax-sm75 sanitize-tile-softmax-ptx-sm75
```

SM70 and CUDA 11.0 receive compile/codegen validation only; SM70 remains
experimental. SM75 runtime exit 77 is SKIP.

A self-checking [GEMM-to-row-softmax example](examples/gemm_row_softmax.cu)
combines the public MMA and softmax tile operations in one warp, writes both
the 16×16 FP32 logits and probabilities, and checks them against an independent
host reference. It is a composition example, not an Attention kernel or a
performance benchmark:

```text
make build-example-row-softmax-sm70 build-example-row-softmax-sm75
make build-example-row-softmax-ptx-sm70 build-example-row-softmax-ptx-sm75
make example-row-softmax-sm75 example-row-softmax-ptx-sm75
```

The SM70 builds are compile/codegen-only; a runtime check requires a selected
SM75 device, and exit 77 is reported as SKIP.

The optional `make bench-row-softmax-sm75` benchmark compares WMMA shared
fallback with PTX m8/m16 register and matched shared-reference paths. It
checks a nontrivial output fingerprint before timing, then reports elapsed
event time divided by completed warps for an identical K=16 staging/MMA prefix,
four alternating rounds and one warp per block. The numbers include that
prefix, output publication, resource effects and scheduling; they are not
isolated softmax latency or end-to-end application performance. It retains
ordinary `expf` and division.

The differential GEMM driver runs six exact-domain cases and ten model-domain
cases: full-mantissa normal-FP16 inputs, including mixed normal exponents, plus
the declared zero cases. It compares the WMMA, PTX m8, and (on SM75) PTX m16
backends. On SM75 it also compares the selected production GEMM path on the
same allocations:

```text
make build-differential-sm70 build-differential-sm75
make build-differential-ptx-sm70 build-differential-ptx-sm75
make test-differential-sm75 test-differential-ptx-sm75
make sanitize-differential-sm75 sanitize-differential-ptx-sm75
```

The exact cases use finite FP16 values q/8 for integer |q| <= 8 and positive,
16-aligned K <= 1024.
Products are exact multiples of 1/64 and the bounded partial sums remain exactly
representable in FP32, which justifies zero-tolerance comparisons for those
fixtures, including direct backend pairs and row-padding sentinels. The general
fixtures retain an exact binary64 reference and use a per-output conditional
model-derived budget; their declared input domain, assumptions, derivation, and
limits are in [the numerical validation model](docs/numerical-validation.md).
This is not a universal NVIDIA accuracy guarantee: the HMMA.1688 extension is
an assumption, arbitrary FP16 inputs are outside the declared domain, and it
does not establish SM70 runtime correctness.

The inline-PTX backends also expose a register-resident row reduction. Because
`nvcuda::wmma::fragment` leaves its lane-to-matrix mapping unspecified, the WMMA
backend can only reduce a row by round-tripping the accumulator through shared
memory; the PTX backends reduce through warp shuffles instead. Both paths combine
the 16 column values as a balanced binary tree in column index order, so they
agree exactly rather than within an error budget:

```text
make build-row-reduce-sm70 build-row-reduce-sm75
make test-row-reduce-sm75
make sanitize-row-reduce-sm75
make bench-row-reduce-sm75
```

The test compares each backend's register-resident result against the
shared-memory reference and against a host evaluation of the declared order, all
with zero tolerance, and additionally against an order-independent exact
reference on the domain where every partial sum is exactly representable in FP32.
It censuses how many additions of the declared tree actually round, so a fixture
whose reduction happens to be exact cannot pass as rounding coverage.
The benchmark reports what it measured for both paths; it is a bounded
microbenchmark of the reduction, not a claim about end-to-end performance.

- SM75 correctness and Compute Sanitizer checks pass on the identified Turing device.
- Compile and codegen checks pass with CUDA 11.0.3 and the local CUDA 12.4 toolkit.
- SM70 is compile-tested and remains experimental until it is run on Volta hardware.
- No performance claims are made for this correctness baseline.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
