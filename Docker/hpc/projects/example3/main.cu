#include <stdio.h>
#include <cuda_runtime.h>

#include <iostream>
#include <memory>

__global__ void checkIndex(void) {
  printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d), gridDim: (%d, %d, %d)\n",
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
}


int main(int argc, char **argv) {
  const int nElem = 32;

  dim3 block(4, 4);
  dim3 grid((nElem + block.x - 1) / block.x, (nElem + block.y - 1) / block.y);

  std::cout << "grid.x" << grid.x << ",grid.y" << grid.y << ",grid.z" << grid.z << std::endl;
  std::cout << "block.x" << block.x << ",block.y" << block.y << ",block.z" << block.z << std::endl;

  checkIndex<<<grid, block>>> ();

  cudaDeviceReset();

  return 0;
}