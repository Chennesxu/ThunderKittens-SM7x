# Numerical validation model

The C2 differential tests use a conditional, model-derived comparison budget
for finite FP16-input, FP32-accumulation GEMM outputs. This is a test contract
for the stated domain. It does not establish behavior for every FP16 input or
prove the internal implementation of a Tensor Core instruction.

## Domain and exact host reference

Each nonzero input is a stored normal binary16 value

```text
x = sign * (1024 + f) * 2^(e - 10),
f in {0, ..., 1023}, e in {-3, ..., 3}, sign in {-1, +1};
```

and zero is also permitted. The reduction dimension `K` is positive, a multiple
of 16, and no greater than 1024. Inputs are multiples of `2^-13`; therefore a
product is a multiple of `2^-26`, has at most 22 significant bits, and is exact
in FP32 and binary64. A partial signed or absolute-product sum has magnitude
below `2^18`, so its scaled integer magnitude requires at most 44 bits. Binary64
has 53 significant bits. Consequently the test keeps, per output cell,

```text
R_ij = sum_k A_ik B_kj
S_ij = sum_k abs(A_ik B_kj)
```

as exact binary64 values for this domain. `R_ij` is not rounded back to FP32.

## Conditional block model

Let a block consume `b >= 1` exact products and an incoming FP32 accumulator
`c`. Put `M = max(abs(c), abs(products))`, `T = abs(c) + sum(abs(products))`,
and `eps = 2^-23`. The model assumes common-exponent alignment changes a term
by less than one grid quantum `q <= eps M`, with at least one maximal-exponent
term unchanged. At most `b` alignments incur loss; aligned terms are summed with
adequate carry width, and a final finite FP32 normalization has error at most
`eps abs(aligned_sum)`. Blocks consume all products exactly once, with no
unaccounted arithmetic stages or overflow.

This permits signed-floor or sign-magnitude alignment behavior. It does not
assert a scalar IEEE rounding mode.

The alignment contribution is at most `b eps M`; accounting for growth before
normalization retains the second-order term:

```text
abs(local_error) <= ((b + 1) eps + b eps^2) T <= gamma(b + 1) T,
gamma(n) = n eps / (1 - n eps).
```

For a chain of nonempty blocks with sizes `b_t`, let `q` be the number of
blocks. Chaining the local inequalities gives `gamma(N) S_ij`, where

```text
N = sum_t (b_t + 1) = K + q <= 2K.
```

For positive `S_ij`, the host helper rounds the gamma quotient upward, then
rounds its product with `S_ij` upward:

```text
gamma_up = nextafter((2K eps) / (1 - 2K eps), +infinity)
E_ij = nextafter(gamma_up S_ij, +infinity).
```

It returns exactly zero for valid zero scale, and rejects invalid metadata or
nonfinite scales. The finite comparator rejects a nonfinite argument or a
negative budget. It accepts a finite output `X` only when
`abs(double(X) - R_ij) <= E_ij`. A pair of finite outputs uses the triangle
bound `abs(double(X) - double(Y)) <= 2 E_ij`. The budget has no absolute floor
and is not based only on `abs(R_ij)`, so cancellation retains its product scale.

## Limits and evidence

The PTX ISA documents at-least-single-precision multiplication and accumulation
but leaves accumulation order, rounding, and subnormal handling unspecified.
The block-chain result is informed by standard gamma analysis, while the
common-exponent alignment model is empirical. The latter paper studies the
four-product HMMA.884 family; applying the generalized block model to
HMMA.1688 is an explicit assumption, not evidence that its internals are the
same. Passing differential cases means only that the tested outputs satisfy this
declared model under its assumptions. It is not an exhaustive hardware accuracy
result, and it does not provide SM70 runtime evidence.

Primary sources:

- NVIDIA, [PTX ISA 7.0, §9.7.13.4.14, printed p. 312](https://docs.nvidia.com/cuda/archive/11.0/pdf/ptx_isa_7.0.pdf#page=326).
- Blanchard, Higham, Lopez, Mary, and Pranesh, [*Mixed Precision Block Fused Multiply-Add*, equations (2.3)–(2.6) and (3.1)–(3.4)](https://eprints.maths.manchester.ac.uk/2750/1/paper.pdf).
- Valpey et al., [*An SMT Formalization of Mixed-Precision Matrix Multiplication*, §§3.1, 4.2–4.3, and 5](https://arxiv.org/html/2502.15999v1).
