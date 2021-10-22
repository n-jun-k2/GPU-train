#pragma once
#include <cuda_runtime.h>

#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <functional>
#include <type_traits>


/**
 * @brief Judgment processing of output result of CUDA api
 *
 * @param err Output result
 * @return
 */
__host__ void CHECK(const cudaError_t err)
{
  if (err == cudaSuccess) return;
  printf("Error: %s:%d, ", __FILE__, __LINE__);
  printf("code: %d, reason: %s\n", err, cudaGetErrorString(err));
  exit(1);
}

/**
 * @brief Create a Device Memory object
 *
 * @tparam T object type.
 * @param size Number of objects.
 * @return Pointer of device memory wrapped with shared_ptr.
 */
template<typename T>
__host__ std::shared_ptr<T> CreateDeviceMemory(const size_t size) {
  static_assert(std::is_pointer<T>::value==false, "pointer support is not available.");
  T* ptr;
  CHECK(cudaMalloc((T**)&ptr, sizeof(T) * size));

  struct _free {
    void operator()(T* p) const{
      CHECK(cudaFree(p));
    }
  };
  return std::shared_ptr<T>(ptr, _free());
}

/**
 * @brief Measures the processing time of the specified function and returns the result in milliseconds
 *
 * @param action Function pointer.
 * @return double
 */
__host__ double elapsedSecondAction(std::function<void()> action) {
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  action();
  end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return elapsed;
}

/**
 * @brief Returns a character string that displays a one-dimensional array as a two-dimensional matrix-like output.
 *
 * @param vec Source vector to display output.
 * @param w Number of dimensions indicating width.
 * @param h Number of dimensions indicating height.
 * @param fill_count Number of display digits.
 * @return std::string.
 */
__host__ std::string printHostMatrix2D(const std::vector<float>& vec, const size_t w, const size_t h, const uint32_t fill_count = 2) {
  std::ostringstream ss;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      ss << std::setw(fill_count) << std::setfill('0') << vec[y * w + x] << ",";
    }
    ss << std::endl;
  }
  return ss.str();
}

/**
 * @brief Returns the index of the best GPU device
 * 
 * @param suitableScore A function that returns the score for optimal conditions
 * @return std::size_t device index.
 */
__host__ std::size_t suitableDeviceIndex(std::function<uint32_t(const cudaDeviceProp&)> suitableScore) {
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
