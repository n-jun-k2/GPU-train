#include <cuda_runtime.h>

#include <iostream>
#include <memory>
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

__global__ void setupDeviceMemory(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = i;
}

int main(int argc, char **argv) {

  const int dev_num = 0;
  const int n_elem = 1 << 24;
  cudaDeviceProp deviceProp;

  CHECK(cudaSetDevice(dev_num));
  CHECK(cudaGetDeviceProperties(&deviceProp, dev_num));

  std::cout << "Using device " << dev_num << ":" <<deviceProp.name << std::endl;


  const int iLen = 512;
  dim3 block(iLen);
  dim3 grid((n_elem + block.x - 1) / block.x);

  std::cout << "Vector size " << n_elem << std::endl;
  std::cout << "Exceution configure <<<" << grid.x << "," << block.x << ">>>" << std::endl;

  std::cout <<" elapsed time : " << elapsedSecondAction([=](){

    std::vector<float> gpuRef_vector(n_elem);
    const uint32_t byte = sizeof(float) * n_elem;
    auto d_vector = createDeviceMemory<float>(byte);

    setupDeviceMemory<<<grid, block>>>(d_vector.get());

    CHECK(cudaMemcpy(gpuRef_vector.data(), d_vector.get(), byte, cudaMemcpyDeviceToHost));
  }) << "'ms" << std::endl;
  CHECK(cudaDeviceReset());
}