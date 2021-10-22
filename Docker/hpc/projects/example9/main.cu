#include "../utils/common.h"

#include <cuda_runtime.h>

#include <iostream>
#include <numeric>
#include <functional>
#include <algorithm>

__host__ std::size_t suitableDevice(std::function<uint32_t(const cudaDeviceProp&)> suitableScore) {
  int max_device_count = 0;
  CHECK(cudaGetDeviceCount(&max_device_count));
  if (max_device_count <= 1) return 0;

  std::vector<cudaDeviceProp> props(max_device_count);
  std::vector<uint32_t> scores(max_device_count);
  uint32_t idx = 0;
  for(auto& prop : props) CHECK(cudaGetDeviceProperties(&prop, idx++));

  idx = 0;
  for(auto& score : scores) score = suitableScore(props[idx++]);

  auto it = std::max_element(std::begin(scores), std::end(scores));
  return std::distance(std::begin(scores), it);
}

int main(int argc, char **argv) {
  std::cout << argv[0] << "Starting ... " << std::endl;

  int device_count;
  CHECK(cudaGetDeviceCount(&device_count));

  if(device_count == 0)
    std::cout << "There are no available device(s) that support CUDA" << std::endl;
  else
    std::cout << "Detected " << device_count << " CUDA capable device(s)" << std::endl;

  const auto use_device = suitableDevice([](const cudaDeviceProp& prop) ->uint32_t {
    return prop.multiProcessorCount;
  });
  CHECK(cudaSetDevice(use_device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, use_device);

  std::cout << "Using Device " << use_device << ":" << prop.name << std::endl;

}