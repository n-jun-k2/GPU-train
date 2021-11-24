#pragma once
#include <cuda_runtime.h>

__device__ float g_f_value;

__global__ void sync_example_kernel() {
  unsigned int id =  blockDim.x * blockIdx.x + threadIdx.x;

  __syncthreads();
  g_f_value = id;
  printf("__syncthreads id (%2d), threadIdx: (%2d, %2d, %2d) blockIdx: (%2d, %2d, %2d) blockDim: (%2d, %2d, %2d), gridDim: (%2d, %2d, %2d) value (%2f)\n",
    id,
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z,
    g_f_value);

  g_f_value = id;
  __threadfence_block();
  printf("__threadfence_block id (%2d), threadIdx: (%2d, %2d, %2d) blockIdx: (%2d, %2d, %2d) blockDim: (%2d, %2d, %2d), gridDim: (%2d, %2d, %2d) value (%2f)\n",
    id,
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z,
    g_f_value);

  g_f_value = id;
  __threadfence_system();
  printf("__threadfence_system id (%2d), threadIdx: (%2d, %2d, %2d) blockIdx: (%2d, %2d, %2d) blockDim: (%2d, %2d, %2d), gridDim: (%2d, %2d, %2d) value (%2f)\n",
    id,
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z,
    g_f_value);


  g_f_value = id;
  __threadfence();
  printf("__threadfence id (%2d), threadIdx: (%2d, %2d, %2d) blockIdx: (%2d, %2d, %2d) blockDim: (%2d, %2d, %2d), gridDim: (%2d, %2d, %2d) value (%2f)\n",
    id,
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z,
    g_f_value);

}