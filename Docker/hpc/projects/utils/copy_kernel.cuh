#pragma
#include <cuda_runtime.h>

template<typename T>
__global__ void copy_kernel(T *out, T *in) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  out[idx] = __ldg(&in[idx]);
}