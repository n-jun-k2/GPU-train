#pragma once
#include <cuda_runtime.h>


template<class T, int __TILE_SIZE__ = 32 >
__global__ void shared_store_load_row_kernel(T *out){

  __shared__ T tile[__TILE_SIZE__][__TILE_SIZE__];

  unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

  tile[threadIdx.y][threadIdx.x] = idx;

  __syncthreads();

  out[idx] = tile[threadIdx.y][threadIdx.x];

}

template<class T, int __TILE_SIZE__ = 32>
__global__ void shared_store_load_col_kernel(T *out) {
  __shared__ T tile[__TILE_SIZE__][__TILE_SIZE__];

  unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

  tile[threadIdx.x][threadIdx.y] = idx;

  __syncthreads();

  out[idx] = tile[threadIdx.x][threadIdx.y];
}

template<class T, int __BDIMY__, int __BDIMX__, int __IPAD__>
__global__ void shared_store_row_load_col_pad_kernel(T *out) {
  __shared__ T tile[__BDIMY__][__BDIMX__ + __IPAD__];

  unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

  tile[threadIdx.y][threadIdx.x] = idx;

  __syncthreads();

  out[idx] = tile[threadIdx.x][threadIdx.y];
}