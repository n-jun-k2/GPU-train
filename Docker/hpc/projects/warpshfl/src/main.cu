#include "../utils/common.h"
#include "../utils/rand_generator_kernel.cuh"
#include "kernel.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <sstream>

void printArray(std::shared_ptr<int> ptr, const int size, const uint32_t fill_count = 2) {
  auto p = ptr.get();
  std::ostringstream ss;
  for (int i = 1; i <= size; ++i)
    ss << std::setw(fill_count) << std::setfill('0') << i << ", " ;
  ss << std::endl;
  for (int i = 0; i < size; ++i)
    ss << std::setw(fill_count) << std::setfill('0') << *(p + i) << ", " ;
  ss << std::endl;
  std::cout << ss.str() << std::endl;
}

int main(int argc, char **argv){

  const auto useDevice = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(useDevice));

  constexpr int bdim = 32;
  constexpr int isize = 1 << 6;
  dim3 block(bdim, 1);
  dim3 grid(isize / block.x, 1);

  std::cout << "(grid, block)(" << grid.x << ", " << block.x << ")" << std::endl;

  auto ref__shfl = createUnifiedMemory<int>(isize);
  auto ref__shfl_up = createUnifiedMemory<int>(isize);
  auto ref__shfl_down = createUnifiedMemory<int>(isize);
  auto ref__shfl_xor = createUnifiedMemory<int>(isize);
  auto ref__shfl__ex1 = createUnifiedMemory<int>(isize);
  auto ref__shfl__ex2 = createUnifiedMemory<int>(isize);

  int warpsize = 32;
  test_shfl_broadcast<<<grid, block>>>(ref__shfl.get(), 16, warpsize);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl " << std::endl;
  printArray(ref__shfl, isize);

  test_shfl_up<<<grid, block>>>(ref__shfl_up.get(), 16, warpsize);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl_up " << std::endl;
  printArray(ref__shfl_up, isize);

  test_shfl_down<<<grid, block>>>(ref__shfl_down.get(), 16, warpsize);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl_down " << std::endl;
  printArray(ref__shfl_down, isize);

  test_shfl_xor<<<grid, block>>>(ref__shfl_xor.get(), 16, warpsize);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl_xor " << std::endl;
  printArray(ref__shfl_xor, isize);


  constexpr unsigned int mask = 0x0000FFFF;
  test_shfl_up<mask><<<grid, block>>>(ref__shfl_up.get(), 16, warpsize);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl_up <" << mask << ">" << std::endl;
  printArray(ref__shfl_up, isize);

  test_shfl_down<mask><<<grid, block>>>(ref__shfl_down.get(), 16, warpsize);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl_down  <" << mask << ">" << std::endl;
  printArray(ref__shfl_down, isize);

  test_shfl_xor<mask><<<grid, block>>>(ref__shfl_xor.get(), 16, warpsize);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl_xor  <" << mask << ">" << std::endl;
  printArray(ref__shfl_xor, isize);

  test_shfl_ex1<<<grid, block>>>(ref__shfl__ex1.get(), 14, 16);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl__ex1 " << std::endl;
  printArray(ref__shfl__ex1, isize);

  test_shfl_ex2<<<grid, block>>>(ref__shfl__ex2.get(), 1, 16);
  CHECK(cudaDeviceSynchronize());
  std::cout << "vecotr ref__shfl__ex2 " << std::endl;
  printArray(ref__shfl__ex2, isize);

  CHECK(cudaDeviceSynchronize());
}