#include "../utils/common.h"
#include "../utils/shared_square_example_kernel.cuh"
#include "../utils/zero_init_kernel.cuh"

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

template<typename T, int COL_SIZE, int ROW_SIZE>
void printSharedPtr2D(const std::shared_ptr<T> ptr){
  auto row_ptr = ptr.get();
  std::cout << "[" << std::endl;
  for(auto col = 0; col < COL_SIZE; ++col) {
    std::cout << " [";
    for(auto row = 0; row < ROW_SIZE; ++row) {
      std::cout << std::setfill('0') << std::setw(4) << *(row_ptr + (ROW_SIZE * col + row)) << "," ;
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "]" << std::endl;
}

int main(int argc, char **argv){

  const auto useDevice = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(useDevice));

  constexpr int TILE_ROW_SIZE = 32;
  constexpr int TILE_COL_SIZE = 32;
  auto gpuRef = createUnifiedMemory<int>(TILE_ROW_SIZE * TILE_COL_SIZE);

  dim3 grid(1);
  dim3 block(TILE_ROW_SIZE, TILE_COL_SIZE);

  auto miliSec = elapsedSecondAction([&](){
    shared_store_load_row_kernel<int><<<grid, block>>>(gpuRef.get());
    CHECK(cudaDeviceSynchronize());
  });

  std::cout << "not confirict time. " << miliSec << " mili sec" << std::endl;
  printSharedPtr2D<int, TILE_COL_SIZE, TILE_ROW_SIZE>(gpuRef);

  for(int i = 0; i < TILE_ROW_SIZE*TILE_COL_SIZE; ++i) {
    *(gpuRef.get() + i) = 0;
  }
  printSharedPtr2D<int, TILE_COL_SIZE, TILE_ROW_SIZE>(gpuRef);

  miliSec = elapsedSecondAction([&](){
    shared_store_load_col_kernel<int><<<grid, block>>>(gpuRef.get());
    CHECK(cudaDeviceSynchronize());
  });
  std::cout << "confirict time. " << miliSec << " mili sec" << std::endl;
  printSharedPtr2D<int, TILE_COL_SIZE, TILE_ROW_SIZE>(gpuRef);

  CHECK(cudaDeviceSynchronize());

}