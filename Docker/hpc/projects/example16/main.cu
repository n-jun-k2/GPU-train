#include "../utils/common.h"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = A[i] + B[i];
}

__host__ void sumArraysOnCPU(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c){
  const auto size = a.size();
  for (size_t i = 0; i < size; ++i)
    c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {

  const auto use_device = suitableDeviceIndex([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(use_device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, use_device);

  std::cout << prop.name << std::endl;
  std::cout << "can map host memory : " << prop.canMapHostMemory << std::endl;

  if (!prop.canMapHostMemory) return 0;


  auto rand_generate = [](auto begin, auto end) {
    std::generate(begin, end, [](){
        static std::random_device seed;
        static std::default_random_engine engine(seed());
        std::uniform_real_distribution<> dist(0, 1.0);
        return dist(engine);
      });
  };

  int ipower = 10;
  if (argc > 1) ipower = atoi(argv[1]);
  int nElem = 1 << ipower;
  size_t nBytes = nElem * sizeof(float);

  if (ipower < 18) {
    std::cout << "Vector size " << nElem << " power " << ipower << " nbytes " << (float)nBytes / (1024.0f) << " KB" << std::endl;
  } else {
    std::cout << "Vector size " << nElem << " power " << ipower << " nbytes " << (float)nBytes / (1024.0f * 1024.0f) << " KB" << std::endl;
  }

  auto h_A = std::vector<float>(nElem);
  auto h_B = std::vector<float>(nElem);
  auto hostRef = std::vector<float>(nElem, 0);
  auto gpuRef = std::vector<float>(nElem, 0);
  auto zeroRef = std::vector<float>(nElem, 0);

  rand_generate(std::begin(h_A), std::end(h_A));
  rand_generate(std::begin(h_B), std::end(h_B));

  sumArraysOnCPU(h_A, h_B, hostRef);
  auto hostSum = std::accumulate(std::begin(hostRef), std::end(hostRef), 0);

  std::cout << "host sum = " << hostSum << std::endl;

  {
    auto d_A = CreateDeviceMemory<float>(static_cast<size_t>(nElem));
    auto d_B = CreateDeviceMemory<float>(static_cast<size_t>(nElem));
    auto d_C = CreateDeviceMemory<float>(static_cast<size_t>(nElem));

    CHECK(cudaMemcpy(d_A.get(), h_A.data(), nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B.get(), h_B.data(), nBytes, cudaMemcpyHostToDevice));

    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    sumArraysZeroCopy<<<grid, block>>>(d_A.get(), d_B.get(), d_C.get(), nElem);

    CHECK(cudaMemcpy(gpuRef.data(), d_C.get(), nBytes, cudaMemcpyDeviceToHost));
    auto deviceSum = std::accumulate(std::begin(gpuRef), std::end(gpuRef), 0);

    std::cout << "device sum = " << deviceSum << std::endl;

    /* zero copy */

    auto zeroHost_A = CreateZeroCopyMemory<float>(nElem);
    auto zeroHost_B = CreateZeroCopyMemory<float>(nElem);

    memcpy(zeroHost_A.get(), h_A.data(), nBytes);
    memcpy(zeroHost_B.get(), h_B.data(), nBytes);

    // float* pd_A;
    // float* pd_B;
    // CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&pd_A), reinterpret_cast<void*>(zeroHost_A.get()), 0));
    // CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&pd_B), reinterpret_cast<void*>(zeroHost_B.get()), 0));
    CHECK(cudaMemcpy(d_C.get(), zeroRef.data(), nBytes, cudaMemcpyHostToDevice));

    // sumArraysZeroCopy<<<grid, block>>>(pd_A, pd_B, d_C.get(), nElem);
    sumArraysZeroCopy<<<grid, block>>>(zeroHost_A.get(), zeroHost_B.get(), d_C.get(), nElem);

    CHECK(cudaMemcpy(zeroRef.data(), d_C.get(), nBytes, cudaMemcpyDeviceToHost));
    auto zeroSum = std::accumulate(std::begin(zeroRef), std::end(zeroRef), 0);

    std::cout << "device zero sum = " << zeroSum << std::endl;

  }
  CHECK(cudaDeviceReset());

  return 0;
}