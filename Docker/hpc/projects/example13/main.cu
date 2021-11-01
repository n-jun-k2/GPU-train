#include "../utils/common.h"

#include <iostream>
#include <cstdlib>

__global__ void nestedHelloWorld(int const iSize, int iDepth) {
  int tid = threadIdx.x;

  printf("Recursion=%d: Hello World from thread %d block %d \n", iDepth, tid, blockIdx.x);

  if (iSize <= 1) return;

  int nthreads = iSize >> 1;

  if(tid==0 && nthreads > 0) {
    nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
    printf("----> nested execution depth: %d \n", iDepth);
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
  {
    nestedHelloWorld<<<1, 10>>>(16, 0);
  }
  CHECK(cudaDeviceReset());
  return 0;
}