#include "tk_sm7x/arch.cuh"

static_assert(tk_sm7x::arch::traits<tk_sm7x::arch::target>::logical_m == 16);
static_assert(tk_sm7x::arch::traits<tk_sm7x::arch::target>::logical_n == 16);
static_assert(tk_sm7x::arch::traits<tk_sm7x::arch::target>::logical_k == 16);

using target_traits = tk_sm7x::arch::traits<tk_sm7x::arch::target>;

static_assert(target_traits::has_mma_m8n8k4);
#if defined(KITTENS_SM70)
static_assert(!target_traits::has_mma_m16n8k8);
static_assert(!target_traits::has_ldmatrix);
#else
static_assert(target_traits::has_mma_m16n8k8);
static_assert(target_traits::has_ldmatrix);
#endif

__global__ void instantiate_device_contract() {}
