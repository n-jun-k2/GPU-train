#include "../utils/common.h"
#include "kernel.cuh"

#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char **argv){

  const auto useDevice = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(useDevice));

  constexpr int bdim = 32;
  constexpr int isize = 1 << 6;
  dim3 block(bdim, 1);
  dim3 grid(isize / block.x, 1);

  helloFromGPU<<<grid, block>>>();

  CHECK(cudaDeviceSynchronize());
}