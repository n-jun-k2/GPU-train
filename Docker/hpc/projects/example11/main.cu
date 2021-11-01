#include "../utils/common.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void thread_synce_test(float *a) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  printf("Hello World from GPU! thread %d : data[0] %f \n", tid, a[0]);
    __syncthreads();
  for (int stride = 0; stride < 1; ++stride) {
    a[0] = tid;
    printf("thread %d :sync data[0] %f \n", tid, a[0]);
    __syncthreads();
    printf("thread %d : worpsize %d  : data[0] %f \n", tid, warpSize >> blockIdx.x, a[0]);
  }

}


int main(int argc, char **argv) {

  const auto use_device = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(use_device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, use_device);

  std::cout << prop.name << std::endl;
  std::cout << "warp size: " << prop.warpSize << std::endl;

  dim3 block(32, 1);
  dim3 grid(2, 1);
  {
    auto d_C = CreateDeviceMemory<float>();
    thread_synce_test <<<grid, block>>>(d_C.get());
  }

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaDeviceReset());

}