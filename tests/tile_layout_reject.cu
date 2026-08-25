#include <cuda_fp16.h>

#include "tk_sm7x/tile.cuh"

struct unsupported_layout {};

__global__ void instantiate_unsupported_layout() {
    __shared__ tk_sm7x::st<__half, 16, 16, unsupported_layout> tile;
    static_cast<void>(tile.data[0]);
}
