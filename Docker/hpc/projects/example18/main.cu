#include "../utils/common.h"
#include "../utils/rand_generator_kernel.cuh"

#include <iostream>
#include <cstdlib>
#include <chrono>


int main(int argc, char **argv){

  const auto use_device = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });

  CHECK(cudaSetDevice(use_device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, use_device);
  std::cout << prop.name << std::endl;

  const int blocksize = 32;
  const size_t nElem = 1 << 24;

  dim3 block(blocksize, 1);
  dim3 grid((nElem + block.x - 1) / block.x, 1);

  auto tp_msec = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
  unsigned long long value = tp_msec.count();

  std::cout << "with array size " << nElem << std::endl;

  auto A = createUnifiedMemory<float>(nElem);

  for(auto i = 0; i < 3; ++i)
    std::cout << "head values :[" << i << "]:" << *(A.get() + i) << std::endl;

  rand_generator_kernel<<<grid, block>>>(value, A.get());
  CHECK(cudaDeviceSynchronize());

  for(auto i = 0; i < 3; ++i)
    std::cout << "head values :[" << i << "]:" << *(A.get() + i) << std::endl;

  CHECK(cudaDeviceSynchronize());

}