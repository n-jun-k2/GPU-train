#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(){
    if (threadIdx.x == 5) printf("Hello World from GPU! thread %d \n", threadIdx.x);
}

int main(int argc, char **argv) {
    printf("Hello World from CPU!\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
}