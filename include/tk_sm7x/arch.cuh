#pragma once

namespace tk_sm7x::arch {

struct sm70 {};
struct sm75 {};

template <class Arch>
struct traits;

template <>
struct traits<sm70> {
    static constexpr int logical_m = 16;
    static constexpr int logical_n = 16;
    static constexpr int logical_k = 16;
    static constexpr int native_m = 8;
    static constexpr int native_n = 8;
    static constexpr int native_k = 4;
    static constexpr bool has_mma_m8n8k4 = true;
    static constexpr bool has_mma_m16n8k8 = false;
    static constexpr bool has_ldmatrix = false;
};

template <>
struct traits<sm75> {
    static constexpr int logical_m = 16;
    static constexpr int logical_n = 16;
    static constexpr int logical_k = 16;
    static constexpr int native_m = 16;
    static constexpr int native_n = 8;
    static constexpr int native_k = 8;
    static constexpr bool has_mma_m8n8k4 = true;
    static constexpr bool has_mma_m16n8k8 = true;
    static constexpr bool has_ldmatrix = true;
};

#if defined(KITTENS_SM70) == defined(KITTENS_SM75)
#error "Define exactly one of KITTENS_SM70 or KITTENS_SM75"
#elif defined(KITTENS_SM70)
using target = sm70;
#else
using target = sm75;
#endif

#if defined(__CUDA_ARCH__)
#if defined(KITTENS_SM70) && __CUDA_ARCH__ != 700
#error "KITTENS_SM70 requires sm_70"
#endif
#if defined(KITTENS_SM75) && __CUDA_ARCH__ != 750
#error "KITTENS_SM75 requires sm_75"
#endif
#endif

}  // namespace tk_sm7x::arch
