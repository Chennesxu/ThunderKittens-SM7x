#include "tk_sm7x/arch.cuh"

static_assert(tk_sm7x::arch::traits<tk_sm7x::arch::target>::logical_m == 16);
static_assert(tk_sm7x::arch::traits<tk_sm7x::arch::target>::logical_n == 16);
static_assert(tk_sm7x::arch::traits<tk_sm7x::arch::target>::logical_k == 16);

__global__ void instantiate_device_contract() {}
