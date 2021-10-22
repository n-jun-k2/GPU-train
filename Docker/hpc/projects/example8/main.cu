#include "../utils/common.h"

#include <cuda_runtime.h>

#include <iostream>
#include <numeric>

__global__ void sumMatrixOnGPU(float *a, float *b, float *c, const int w, const int h) {
  const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const unsigned int idx = iy * w + ix;

  if (ix < w && iy < h)
    c[idx] = a[idx] + b[idx];
}


int main(int argc, char **argv) {
  const int deviceIdx = 0;
  cudaDeviceProp deviceProp;


  CHECK(cudaGetDeviceProperties(&deviceProp, deviceIdx));
  std::cout << "Using Device " << deviceIdx << ":" << deviceProp.name << std::endl;
  CHECK(cudaSetDevice(deviceIdx));

  const int width   = 1 << 9;
  const int heigth  = 1 << 9;
  const int bytes   = sizeof(float) * width * heigth;

  std::cout << "matrix size :" << width << " x " << heigth  << std::endl;

  std::vector<float> mtx_A(width * heigth);
  std::vector<float> mtx_B(width * heigth);
  std::vector<float> mtx_C(width * heigth);
  std::iota(std::begin(mtx_A), std::end(mtx_A), 1);
  std::iota(std::begin(mtx_B), std::end(mtx_B), 1);

  auto elapsed_time = elapsedSecondAction([&](){

    auto d_mtx_a = CreateDeviceMemory<float>(bytes);
    auto d_mtx_b = CreateDeviceMemory<float>(bytes);
    auto d_mtx_c = CreateDeviceMemory<float>(bytes);
    CHECK(cudaMemcpy(d_mtx_a.get(), mtx_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mtx_b.get(), mtx_B.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mtx_c.get(), mtx_C.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (heigth + block.y - 1) / block.y);
    sumMatrixOnGPU<<<grid, block>>>(d_mtx_a.get(), d_mtx_b.get(), d_mtx_c.get(), width, heigth);
    std::cout << "sumMatrixOnGPU <<< (" << grid.x << "," << grid.y << "), " << "(" << block.x << "," << block.y << ") >>>" << std::endl;

    CHECK(cudaMemcpy(mtx_C.data(), d_mtx_c.get(), bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
  });

  std::cout << "time :" << elapsed_time << "'ms" << std::endl;

  CHECK(cudaDeviceReset());

}