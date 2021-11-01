#include "../utils/common.h"

#include <iostream>
#include <cstdlib>

__device__ float g_devData;

__constant__ float g_devSymbol_Data;

__global__ void checkGlobalVariable() {
  printf("Device: the value of the global variable is %f\n", g_devData);
  printf("Device: the value of the constant variable is %f\n", g_devSymbol_Data);
  g_devData += 4.0f;
}

/**
* [maxThreadsPerBlock]:カーネルが起動するブロック１つあたりのスレッドの最大数を指定します。
* [minBlockPerMultiprocessor]:省略可能。SMに割り当てるスレッドブロックの最小数として望ましい値を指定する。
＊コンパイル単位内の全てのカーネルによって使用されるレジスタの最大数を制御する場合は、-maxrregcountコンパイラオプションを使用する。
*/
__global__ void __launch_bounds__(8, 1) kernelFunc1() {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  printf("%2d,%2d : HELLO DEVICE WORLD\n", x, y);
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
  {
    float value = 3.14f;

    CHECK(cudaMemcpyToSymbol(g_devData, &value, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(g_devSymbol_Data, &value, sizeof(float)));

    printf("Host: copied %f to the global variable \n", value);

    checkGlobalVariable<<<1, 1>>>();

    CHECK(cudaMemcpyFromSymbol(&value, g_devData, sizeof(float)));

    printf("Host: copied %f to the global variable \n", value);

    value = 3.14f;
    float* p_g_devData;
    CHECK(cudaGetSymbolAddress((void**)&p_g_devData, g_devData));
    CHECK(cudaMemcpy(p_g_devData, &value, sizeof(float), cudaMemcpyHostToDevice));

    checkGlobalVariable<<<1, 1>>>();

    kernelFunc1<<<1, 1>>>();
    kernelFunc1<<<1, 8>>>();
    dim3 block(8, 1);
    kernelFunc1<<<2, block>>>();
    dim3 block_b(8, 2);
    kernelFunc1<<<1, block_b>>>(); // NG
  }
  CHECK(cudaDeviceReset());
  return 0;
}