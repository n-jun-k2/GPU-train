#include "../utils/common.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void warmingup(float *a) {

}

int main(int argc, char **argv) {

  const auto use_device = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(use_device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, use_device);

  int size = 64;
  int block_size = 64;
  if (argc > 1 ) block_size = std::atoi(argv[1]);
  if (argc > 2) size = std::atoi(argv[2]);
  {
    /* device resource */

  }
  CHECK(cudaDeviceReset());

}