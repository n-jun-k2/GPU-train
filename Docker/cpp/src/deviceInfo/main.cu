#include "../utils/common.h"


int main(int argc, char **argv){

    printf("%s Starting ...\n", argv[0]);

    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    if(deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }else {
        printf("Detected %d CUDA capable device(s)\n", deviceCount);
    }

    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    CHECK(cudaSetDevice(dev));//0番目のデバイスに命令をセットすることができる。(デフォルトは0番目)
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: \"%s\"\n", dev, prop.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf(" CUDA Driver Version / Runtime Version\t%d.%d / %d.%d\n",
            driverVersion / 1000, (driverVersion % 100) / 10,
            runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    printf(" CUDA Capability Major/Minor version number:\t%d.%d\n",
            prop.major, prop.minor);

    printf(" Total amount of global memory:\t %.2f MBytes (%llu bytes)\n",
            (float)prop.totalGlobalMem / pow(1024.0, 3),
            (unsigned long long)prop.totalGlobalMem);

    printf(" GPU Clock rate:\t%.0f MHz (%0.2f GHz)\n",
            prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);
            printf("  Memory Clock rate:                             %.0f Mhz\n",
            prop.memoryClockRate * 1e-3f);
    printf(" Memory Bus Width:\t%d-bit\n",
            prop.memoryBusWidth);

    if (prop.l2CacheSize){
        printf(" L2 Cache Size:\t%d bytes\n",
                prop.l2CacheSize);
    }

    printf("  Max Texture Dimension Size (x,y,z)\t1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            prop.maxTexture1D, prop.maxTexture2D[0],
            prop.maxTexture2D[1], prop.maxTexture3D[0],
            prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers\t1D=(%d) x %d, 2D=(%d,%d) x %d\n",
            prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1],
            prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1],
            prop.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory:\t%lu bytes\n",
            prop.totalConstMem);
    printf("  Total amount of shared memory per block:\t%lu bytes\n",
            prop.sharedMemPerBlock);
    printf("  Total number of registers available per block:\t%d\n",
            prop.regsPerBlock);
    printf("  Warp size:\t%d\n",
            prop.warpSize);
    printf("  Maximum number of threads per multiprocessor:\t%d\n",
            prop.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:\t%d\n",
            prop.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:\t%d x %d x %d\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:\t%d x %d x %d\n",
            prop.maxGridSize[0], prop.maxGridSize[1],
            prop.maxGridSize[2]);
    printf("  Maximum memory pitch:\t%lu bytes\n",
            prop.memPitch);

    return 0;
}