#include "../utils/common.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>


template<unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n) {
  const unsigned int LOOP_UNROLLING = 8;
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * LOOP_UNROLLING + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x * LOOP_UNROLLING;

  if (idx + (LOOP_UNROLLING - 1) * blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + blockDim.x * 2];
    int a4 = g_idata[idx + blockDim.x * 3];
    int b1 = g_idata[idx + blockDim.x * 4];
    int b2 = g_idata[idx + blockDim.x * 5];
    int b3 = g_idata[idx + blockDim.x * 6];
    int b4 = g_idata[idx + blockDim.x * 7];
    g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
  }

  __syncthreads();

  if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
  __syncthreads();

  if (iBlockSize >= 512 && tid < 256) idata[tid] += idata[tid + 256];
  __syncthreads();

  if (iBlockSize >= 256 && tid < 128) idata[tid] += idata[tid + 128];
  __syncthreads();

  if (iBlockSize >= 128 && tid < 64) idata[tid] += idata[tid + 64];
  __syncthreads();

  if (tid < warpSize) {
    volatile int *vsmem = idata;
    vsmem[tid] += vsmem[tid + warpSize];
    vsmem[tid] += vsmem[tid + (warpSize >> 1)];
    vsmem[tid] += vsmem[tid + (warpSize >> 2)];
    vsmem[tid] += vsmem[tid + (warpSize >> 3)];
    vsmem[tid] += vsmem[tid + (warpSize >> 4)];
    vsmem[tid] += vsmem[tid + (warpSize >> 5)];
  }

  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char **argv) {

  const auto use_device = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(use_device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, use_device);

  std::cout << prop.name << std::endl;
  std::cout << "warp size: " << prop.warpSize << std::endl;

  const int size = 1<<10;
  std::cout << "\twith array size " << size << std::endl;
  std::vector<int> h_idata(size);
  std::vector<int> h_odata(size);
  std::vector<int> temp(size);


  auto rand_generate = [](std::vector<int>::iterator begin, std::vector<int>::iterator end) {
    std::generate(begin, end, [](){
        static std::random_device seed;
        static std::default_random_engine engine(seed());
        std::uniform_int_distribution<> dist(0, 255);
        return dist(engine);
      });
  };
  rand_generate(std::begin(h_idata), std::end(h_idata));
  std::copy(std::begin(h_idata), std::end(h_idata), std::begin(temp));

  {
    dim3 block(512, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    std::cout << "grid " << grid.x << " block " << block.x << std::endl;

    auto d_idata = createDeviceMemory<int>(size);
    auto d_odata = createDeviceMemory<int>(size);

    int cpu_sum = 0;
    auto cpu_sum_time = elapsedSecondAction([&](){
      cpu_sum = std::accumulate(std::begin(temp), std::end(temp), 0);
    });
    std::cout << "cpu reduce \telapsed "<< cpu_sum_time << " mili sec cpu_sum: " << cpu_sum << std::endl;

    CHECK(cudaMemcpy(d_idata.get(), h_idata.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    auto gpu_sum_time = elapsedSecondAction([&](){
      reduceCompleteUnroll<1024><<<grid, block>>>(d_idata.get(), d_odata.get(), size);
      CHECK(cudaDeviceSynchronize());
    });
    CHECK(cudaMemcpy(h_odata.data(), d_odata.get(), sizeof(int) * size, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    auto gpu_sum = std::accumulate(std::begin(h_odata), std::end(h_odata), 0);
    std::cout << "gpu neighbored elapsed "<< gpu_sum_time << " mili sec gpu_sum :" << gpu_sum << std::endl;

  }
  CHECK(cudaDeviceReset());
}