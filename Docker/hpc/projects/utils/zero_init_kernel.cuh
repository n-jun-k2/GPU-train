#pragma once
#include <cuda_runtime.h>

template<typename T>
__global__ void zero_init_kernel(T *mem) {
  unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  mem[id] = 0;
}