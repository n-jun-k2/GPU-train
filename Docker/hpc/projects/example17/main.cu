#include "../utils/common.h"
#include "../utils/rand_generator_kernel.cuh"
#include "../utils/zero_init_kernel.cuh"
#include "../utils/read_offset_kernel.cuh"

#include <iostream>
#include <cstdlib>

int main(int argc, char **argv) {

  const auto use_device = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });

  CHECK(cudaSetDevice(use_device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, use_device);
  std::cout << prop.name << std::endl;

  const int nElem = 1 << 24;
  std::cout << "with array size " << nElem << std::endl;

  int offset = 0;
  int blocksize = 32;
  if (argc > 1) offset = atoi(argv[1]);
  if (argc > 2) blocksize = atoi(argv[2]);

  dim3 block(blocksize, 1);
  dim3 grid((nElem + block.x - 1) / block.x, 1);

  auto tp_msec = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
  unsigned long long value = tp_msec.count();


  auto devA = CreateDeviceMemory<float>(nElem);
  auto devB = CreateDeviceMemory<float>(nElem);
  auto devC = CreateDeviceMemory<float>(nElem);

  rand_generator_kernel<<<block, grid>>>(value, devA.get());
  rand_generator_kernel<<<block, grid>>>(value, devB.get());
  zero_init_kernel<float><<<block, grid>>>(devC.get());

  CHECK(cudaDeviceSynchronize());

  /* check device memory
  auto pin = CreatePinedMemory<float>(nElem);
  CHECK(cudaMemcpy(pin.get(), devA.get(), sizeof(float) * nElem, cudaMemcpyDeviceToHost));
  for(auto i = 0; i < 3; ++i)
    std::cout << "head values :[" << i << "]:" << *(pin.get() + i) << std::endl;
  */

  auto elapsed = elapsedSecondAction([&](){
    read_offset_kernel<float><<<block, grid>>>(devA.get(), devB.get(), devC.get(), nElem, offset);
    CHECK(cudaDeviceSynchronize());
  });

  std::cout << "read_offset_kernel <<<" << grid.x << "," << block.x << ">>> offset :" << offset << " elapsed " << elapsed << "mili sec" << std::endl;

  CHECK(cudaDeviceSynchronize());

}