#include <cuda_runtime.h>

#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <string>
#include <sstream>


__host__ void CHECK(const cudaError_t err)
{
  if (err == cudaSuccess) return;
  printf("Error: %s:%d, ", __FILE__, __LINE__);
  printf("code: %d, reason: %s\n", err, cudaGetErrorString(err));
  exit(1);
}


template<typename T>
__host__ std::shared_ptr<T> createDeviceMemory(const size_t byte_size) {
  static_assert(std::is_pointer<T>::value==false, "pointer support is not available.");
  T* ptr;
  CHECK(cudaMalloc((T**)&ptr, byte_size));

  struct _free {
    void operator()(T* p) const{
      CHECK(cudaFree(p));
    }
  };
  return std::shared_ptr<T>(ptr, _free());
}

__host__ std::string printHostMatrix(const std::vector<float>& vec, const size_t w, const size_t h) {
  std::ostringstream ss;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      ss << std::setw(2) << std::setfill('0') << vec[y * w + x] << ",";
    }
    ss << std::endl;
  }
  return ss.str();
}

__global__ void  printThreadIndex(float* A, const int w, const int h) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = iy * h + ix;

  printf("thread id (%d, %d) block id (%d, %d) coordinate (%d, %d) global index %2d ival %2.1f\n",
    threadIdx.x, threadIdx.y,
    blockIdx.x, blockIdx.y,
    ix, iy, idx, A[idx]);
}

int main(int argc, char **argv) {
  const int deviceIdx = 0;
  cudaDeviceProp deviceProp;

  CHECK(cudaGetDeviceProperties(&deviceProp, deviceIdx));
  std::cout << "Using Device " << deviceIdx << ":" << deviceProp.name << std::endl;
  CHECK(cudaSetDevice(deviceIdx));


  const int width = 8;
  const int heigth = 6;
  const int bytes = sizeof(float) * width * heigth;

  std::vector<float> mtx_A(width * heigth);
  std::iota(std::begin(mtx_A), std::end(mtx_A), 1);
  std::cout << std::endl << printHostMatrix(mtx_A, width, heigth) << std::endl;

  {
    auto d_mtx_a = createDeviceMemory<float>(bytes);
    CHECK(cudaMemcpy(d_mtx_a.get(), mtx_A.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(4, 2);
    dim3 grid((width + block.x - 1) / block.x, (heigth + block.y - 1) / block.y);
    printThreadIndex<<< grid, block >>>(d_mtx_a.get(), width, heigth);
    CHECK(cudaDeviceSynchronize());
  }

  CHECK(cudaDeviceReset());

}