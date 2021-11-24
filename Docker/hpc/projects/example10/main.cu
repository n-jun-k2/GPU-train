#include "../utils/common.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void warmingup(float *a) {

}

__global__ void mathKernel1(float *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float ia, ib;
  ia = ib = 0.0f;

  if(tid % 2 == 0) {
    ia = 100.0f;
  } else {
    ib = 200.0f;
  }
  a[tid] = ia + ib;
}

__global__ void mathKernel2(float *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float ia, ib;
  ia = ib = 0.0f;

  if ((tid / warpSize) % 2 == 0) {
    ia = 100.0f;
  }else{
    ib = 200.0f;
  }
  a[tid] = ia + ib;
}

int main(int argc, char **argv) {

  const auto use_device = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(use_device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, use_device);

  std::cout << "Using Device " << use_device << ":" << prop.name << std::endl;

  int size = 64;
  int block_size = 64;
  if (argc > 1 ) block_size = std::atoi(argv[1]);
  if (argc > 2) size = std::atoi(argv[2]);
  {
    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x, 1);
    auto d_C = createDeviceMemory<float>(size);

    std::cout << "Execution configure (block " << block.x << " " << grid.x << ")" << std::endl;
    auto kernel_time = elapsedSecondAction([&](){
      warmingup<<<grid, block>>>(d_C.get());
      CHECK(cudaDeviceSynchronize());
    });
    std::cout << "warmup <<<" << grid.x << "," << block.x << " >>> elapsed " << kernel_time << " milisec"  << std::endl;
    CHECK(cudaGetLastError());

    kernel_time = elapsedSecondAction([&]() {
      mathKernel1<<<grid, block>>>(d_C.get());
      CHECK(cudaDeviceSynchronize());
    });
    std::cout << "mathKernel1 <<<" << grid.x << "," << block.x << ">>> elapsed " << kernel_time << " milisec" << std::endl;
    CHECK(cudaGetLastError());

    kernel_time = elapsedSecondAction([&]() {
      mathKernel2<<<grid, block>>>(d_C.get());
      CHECK(cudaDeviceSynchronize());
    });
    std::cout << "mathKernel2 <<<" << grid.x << "," << block.x << ">>> elapsed " << kernel_time << " milisec" << std::endl;
    CHECK(cudaGetLastError());
  }
  CHECK(cudaDeviceReset());

}