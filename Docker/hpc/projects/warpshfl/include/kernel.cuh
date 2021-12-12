#pragma once
#include <cuda_runtime.h>

__global__ void printLaneIDandWarpID(){
  int laneID = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;
  printf("Hello World from GPU! thread %d lane %d warp %d\n", threadIdx.x, laneID, warpID);
}

template<unsigned int mask = 0>
__global__ void test_shfl_broadcast(int *out, const int delta, unsigned int width) {
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int value = __shfl_sync(0, threadIdx.x, delta, width);
  out[idx] = value;
}

template<unsigned int mask = 0xFFFFFFFF>
__global__ void test_shfl_up(int *out, const int delta, unsigned int width) {
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int value = __shfl_up_sync(mask, threadIdx.x, delta, width);
  out[idx] = value;
}

template<unsigned int mask = 0xFFFFFFFF>
__global__ void test_shfl_down(int *out, const int delta, unsigned int width) {
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int value = __shfl_down_sync(mask, threadIdx.x, delta, width);
  out[idx] = value;
}

template<unsigned int mask = 0xFFFFFFFF>
__global__ void test_shfl_xor(int *out, const int delta, unsigned int width) {
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int value = __shfl_xor_sync(mask, threadIdx.x, delta, width);
  out[idx] = value;
}


template<unsigned int mask = 0xFFFFFFFF>
__global__ void test_shfl_ex1(int *out, const int delta, unsigned int width) {
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int value = __shfl_up_sync(mask, threadIdx.x, delta, width);
  value += __shfl_down_sync(mask, threadIdx.x,  2 , width);;
  out[idx] = value;
}

template<unsigned int mask = 0xFFFFFFFF>
__global__ void test_shfl_ex2(int *out, const int delta, unsigned int width) {
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int value = __shfl_xor_sync(mask, threadIdx.x, delta, width);
  value += __shfl_xor_sync(mask, value, delta, width);;
  out[idx] = value;
}