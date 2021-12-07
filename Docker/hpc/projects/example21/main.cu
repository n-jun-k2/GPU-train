#include "../utils/common.h"
#include "../utils/constant_example_kernel.cuh"
#include "../utils/rand_generator_kernel.cuh"
#include "../utils/zero_init_kernel.cuh"

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cmath>

int main(int argc, char **argv){

  const auto useDevice = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(useDevice));

  constexpr int radius = 4;
  constexpr int bdim = 32;
  constexpr int isize = 1 << 6;
  constexpr int size = (isize) + (radius * 2);
  dim3 block(bdim, 1);
  dim3 grid(isize / block.x, 1);
  dim3 block_init(bdim, 1);
  dim3 grid_init((size / block.x) + std::ceil((size % block.x) / float(block.x)), 1);


  std::cout << "array size : " << size << std::endl;
  std::cout << "(grid, block) " << grid.x << "," << block.x << std::endl;
  std::cout << " init (grid, block) " << grid_init.x << "," << block_init.x  << std::endl;

  // setup cmem <constat memory>
  constexpr float constnatList[] = {0.0f, 0.8f, -0.2f, 0.03809f, -0.00357f};
  CHECK(cudaMemcpyToSymbol(cmem, constnatList, (radius + 1)  * sizeof(float)));

  auto d_in = createUnifiedMemory<float>(size);
  auto d_out = createUnifiedMemory<float>(size);

  unsigned long long rand_seed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  rand_generator_kernel<<<grid_init, block_init>>>(rand_seed, d_in.get());
  zero_init_kernel<float><<<grid_init, block_init>>>(d_out.get());
  CHECK(cudaDeviceSynchronize());

  constant_example_kernel<bdim, radius><<<grid, block>>>(d_in.get(), d_out.get());
  CHECK(cudaDeviceSynchronize());


  std::cout << "d_in = [";
  for(int i = 0; i < size; ++i) {
    std::cout << *(d_in.get() + i) << ",";
  }
  std::cout << "]" << std::endl;

  std::cout << "d_out = [";
  for(int i = 0; i < size; ++i) {
    std::cout << *(d_out.get() + i) << ",";
  }
  std::cout << "]" << std::endl;

}