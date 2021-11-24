#include "../utils/common.h"
#include "../utils/rand_generator_kernel.cuh"
#include "../utils/sync_example_kernel.cuh"

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <string>

std::string sharedMemConfigToString(cudaSharedMemConfig config){
  switch (config)
  {
  case cudaSharedMemBankSizeDefault:
    return std::string("cudaSharedMemBankSizeDefault");
  case cudaSharedMemBankSizeEightByte:
    return std::string("cudaSharedMemBankSizeEightByte");
  case cudaSharedMemBankSizeFourByte:
    return std::string("cudaSharedMemBankSizeFourByte");
  }
  return std::string();
}

int main(int argc, char **argv){

  const auto useDevice = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });

  CHECK(cudaSetDevice(useDevice));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, useDevice));
  std::cout << prop.name << std::endl;

  /*Access mode*/
  cudaSharedMemConfig sharedConfig;
  CHECK(cudaDeviceGetSharedMemConfig(&sharedConfig));
  std::cout << "get is " << sharedMemConfigToString(sharedConfig) << std::endl;

  CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  CHECK(cudaDeviceGetSharedMemConfig(&sharedConfig));
  std::cout << "set is " << sharedMemConfigToString(sharedConfig) << std::endl;

  /* capacity setting */
  cudaFuncCache funcCache = cudaFuncCachePreferNone;
  CHECK(cudaDeviceSetCacheConfig(funcCache));

  const int blocksize = 32;
  const size_t nElem = 1 << 6;

  dim3 block(blocksize, 1);
  dim3 grid((nElem + block.x - 1) / block.x, 1);

  sync_example_kernel<<<grid, block>>>();
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaDeviceReset());

}