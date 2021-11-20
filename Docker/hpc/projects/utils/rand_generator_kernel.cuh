#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>


__global__ void rand_generator_kernel(unsigned long long seed, float *mem) {
  unsigned int id =  blockDim.x * blockIdx.x + threadIdx.x;
  curandState randState;

  curand_init(seed, id, 0, &randState);

  mem[id] = curand_uniform(&randState);
}