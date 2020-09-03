#include "../utils/common.h"
#include <stdlib.h>

__global__ void mathKernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    //偶数スレッドと奇数スレッドでデータを分割する
    if (tid % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel2(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    //ワープ事にデータを分割する
    if ((tid / warpSize) % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

void run_kernel(const dim3& grid, const dim3& block, float (*d_c), void(*global_func)(float*)) {
    size_t iStart, iElapse;
    iStart = cpuSecond();
    global_func<<<grid, block>>>(d_c);
    CHECK(cudaDeviceSynchronize());
    iElapse = cpuSecond() - iStart;
    printf("kernel func <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x, (int)iElapse);
    CHECK(cudaGetLastError());
}


int main(int argc, char **argv) {

    int dev = 0;// デバイスのセットアップ
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device %d: %s \n", argv[0], dev, deviceProp.name);

    //データサイズの設定
    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);

    //実行設定のセットアップ
    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d) \n", block.x, grid.x);

    //GPUメモリを確保
    float *d_c;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float**) &d_c, nBytes));
    
    //オーバーヘッドを取り除くためにwarmingupカーネルを実行
    size_t iStart, iElapse;
    CHECK(cudaDeviceSynchronize());

    iStart = cpuSecond();
    warmingup<<<grid, block>>>(d_c);
    iElapse = cpuSecond() - iStart;
    printf("warmup <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x, (int)iElapse);
    CHECK(cudaGetLastError());


    // _/_/_/_/_/_/_/_/_/_/run kernels /_/_/_/_/_/_/_/_/_/_/_/_/_/
    run_kernel(grid, block, d_c, mathKernel1);
    run_kernel(grid, block, d_c, mathKernel2);

    CHECK(cudaFree(d_c));
    CHECK(cudaDeviceReset());

    return 0;
}