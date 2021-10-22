#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <memory>
#include <functional>
#include <type_traits>
#include <cmath>

__host__ void CHECK(const cudaError_t err)
{
  if (err == cudaSuccess) return;
  printf("Error: %s:%d, ", __FILE__, __LINE__);
  printf("code: %d, reason: %s\n", err, cudaGetErrorString(err));
  exit(1);
}

__host__ void CheckResult(std::vector<float>& host, std::vector<float>& device) {
  const double eps = 1.0E-8;
  bool is_match = false;
  for(int i = 0; i < host.size(); ++i) {
    is_match = std::abs(host[i] - device[i]) <= eps;
    if (!is_match) break;
  }

  if (is_match) std::cout << "arrays match." << std::endl;
  else std::cout <<"arrays do not match." << std::endl;
}

__host__ void SumArraysOnCPU(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c){
  const auto size = a.size();
  for (size_t i = 0; i < size; ++i)
    c[i] = a[i] + b[i];
}

__global__ void SumArraysOnGPU(float *a, float *b, float *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template<typename T>
__host__ std::shared_ptr<T> CreateDeviceMemory(const size_t byte_size) {
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


int main(int argc, char  **argv) {

  const int dev_number = 0;
  CHECK(cudaSetDevice(dev_number));

  static const int element_size = 32;

  using host_type = std::vector<float>;
  host_type h_A_vector(element_size);
  host_type h_B_vector(element_size);
  host_type h_C_vector(element_size);
  host_type h_gpuRef_vector(element_size);

  auto rand_generate = [](host_type::iterator begin, host_type::iterator end) {
    std::generate(begin, end, [](){
        static std::random_device seed;
        static std::default_random_engine engine(seed());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        return dist(engine);
      });
  };

  rand_generate(std::begin(h_A_vector), std::end(h_A_vector));
  rand_generate(std::begin(h_B_vector), std::end(h_B_vector));
  {
    const uint32_t byte_size = sizeof(float) * element_size;
    auto d_A_vector = CreateDeviceMemory<float>(byte_size);
    auto d_B_vector = CreateDeviceMemory<float>(byte_size);
    auto d_C_vector = CreateDeviceMemory<float>(byte_size);

    CHECK(cudaMemcpy(d_A_vector.get(), h_A_vector.data(), byte_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B_vector.get(), h_B_vector.data(), byte_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C_vector.get(), h_C_vector.data(), byte_size, cudaMemcpyHostToDevice));

    dim3 block(element_size);
    dim3 grid(1);

    SumArraysOnGPU<<<grid, block>>>(d_A_vector.get(), d_B_vector.get(), d_C_vector.get());
    std::cout << "Exceution configure <<<" << grid.x << "," << block.x << ">>>" << std::endl;
    CHECK(cudaMemcpy(h_gpuRef_vector.data(), d_C_vector.get(), byte_size, cudaMemcpyDeviceToHost));
    SumArraysOnCPU(h_A_vector, h_B_vector, h_C_vector);

    CheckResult(h_C_vector, h_gpuRef_vector);

  }
  CHECK(cudaDeviceReset());

  return 0;
}