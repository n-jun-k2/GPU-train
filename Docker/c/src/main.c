#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFrom(){
    printf("Hello World from GPU!\n")
}

int main(){
    printf("Hello World from CPU!\n")
}