#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(){
    printf("Hello World from GPU!\n");
}

int main( int argc, char **argv)
{
    printf("Hello World from CPU!\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}