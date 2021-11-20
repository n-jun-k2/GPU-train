#pragma once
#include <cuda_runtime.h>

template<typename T>
__global__ void read_offset_kernel(T *A, T *B, T *C, const int n, int offset) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = i + offset;

  if (k < n) C[i] = A[k] + B[k];
}