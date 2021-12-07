#pragma once
#include <cuda_runtime.h>

__global__ void helloFromGPU(){
  printf("Hello World from GPU! thread %d \n", threadIdx.x);
}