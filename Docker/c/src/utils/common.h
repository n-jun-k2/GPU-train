#ifndef __COMMON_FILE_H
#define __COMMON_FILE_H

#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(call) {                                                       \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess) {                                             \
        printf("Error: %s:%d, \n", __FILE__, __LINE__);                     \
        printf("code:%d, reason: %s \n", error, cudaGetErrorString(error)); \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \



double cpuSecond(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif