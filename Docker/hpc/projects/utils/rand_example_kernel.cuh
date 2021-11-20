#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ static unsigned long value = 0;

__global__ void  rand_kernel(unsigned long long seed) {
  unsigned long id = threadIdx.x + blockDim.x * blockIdx.x;
  curandState randState;

  curand_init(seed, id, 0, &randState);

  __syncthreads();
  float randValue = curand_uniform(&randState);

  value = id;
  printf("id (%2d), seed(%lu), threadIdx: (%2d, %2d, %2d) blockIdx: (%2d, %2d, %2d) blockDim: (%2d, %2d, %2d), gridDim: (%2d, %2d, %2d) value(%2lu) rand(%f)\n",
    id, seed,
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z,
    value, randValue);
}