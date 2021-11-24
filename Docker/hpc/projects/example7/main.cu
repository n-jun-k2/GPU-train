#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <type_traits>
#include <numeric>
#include <vector>
#include <chrono>
#include <functional>

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

__host__ double elapsedSecondAction(std::function<void()> action) {
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  action();
  end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return elapsed;
}

__host__ std::string printHostMatrix(const std::vector<float>& vec, const size_t w, const size_t h) {
  std::ostringstream ss;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      ss << std::setw(4) << std::setfill('0') << vec[y * w + x] << ",";
    }
    ss << std::endl;
  }
  return ss.str();
}

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

  const int width   = 1 << 14;
  const int heigth  = 1 << 14;
  const int bytes   = sizeof(float) * width * heigth;

  std::cout << "matrix size :" << width << " x " << heigth  << std::endl;

  std::vector<float> mtx_A(width * heigth);
  std::vector<float> mtx_B(width * heigth);
  std::vector<float> mtx_C(width * heigth);
  std::iota(std::begin(mtx_A), std::end(mtx_A), 1);
  std::iota(std::begin(mtx_B), std::end(mtx_B), 1);

  auto elapsed_time = elapsedSecondAction([&](){

    auto d_mtx_a = createDeviceMemory<float>(bytes);
    auto d_mtx_b = createDeviceMemory<float>(bytes);
    auto d_mtx_c = createDeviceMemory<float>(bytes);
    CHECK(cudaMemcpy(d_mtx_a.get(), mtx_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mtx_b.get(), mtx_B.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mtx_c.get(), mtx_C.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (heigth + block.y - 1) / block.y);
    sumMatrixOnGPU<<<grid, block>>>(d_mtx_a.get(), d_mtx_b.get(), d_mtx_c.get(), width, heigth);
    std::cout << "sumMatrixOnGPU <<< (" << grid.x << "," << grid.y << "), " << "(" << block.x << "," << block.y << ") >>>" << std::endl;

    CHECK(cudaMemcpy(mtx_C.data(), d_mtx_c.get(), bytes, cudaMemcpyDeviceToHost));
    // std::cout << std::endl << printHostMatrix(mtx_C, width, heigth) << std::endl;
    CHECK(cudaDeviceSynchronize());
  });

  std::cout << "time :" << elapsed_time << "'ms" << std::endl;

  CHECK(cudaDeviceReset());

}