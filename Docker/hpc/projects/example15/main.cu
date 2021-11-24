#include "../utils/common.h"

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
  std::cout << "can map host memory : " << prop.canMapHostMemory << std::endl;

  {
    const unsigned int isize = 1 << 22;
    const unsigned int nbyte = isize * sizeof(float);
    auto pin_a = createPinedMemory<float>(isize);
    auto dev_a = createDeviceMemory<float>(isize);

    memset(pin_a.get(), 0, nbyte);

    for (int i = 0; i < isize; ++i) *(pin_a.get() + 1) = 100.0f;

    CHECK(cudaMemcpy(dev_a.get(), pin_a.get(), nbyte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(pin_a.get(), dev_a.get(), nbyte, cudaMemcpyDeviceToHost));

  }
  CHECK(cudaDeviceReset());

  return 0;
}