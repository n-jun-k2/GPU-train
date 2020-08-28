# Steps to build Docker for GPU on Ubuntu on WSL2
## reference
- [WSL command reference](https://docs.microsoft.com/ja-jp/windows/wsl/reference)
- [WSL2 update method](https://qiita.com/naka345/items/eba1870fba589a68847e#wsl2%E4%B8%8A%E3%81%ABdocker%E3%82%92%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
- [Nvidia driver for WSL](https://developer.nvidia.com/cuda/wsl/download)

## Prerequisite
- OS: Windows 10 Pro Insider Preview (OS Build 2015 or later)
- Linux kernel >=4.19.121 (Linux running on WSL)
- Nvidia driver installed for WSL

## Execute each script.
Execute in the following order.

1. wsl-install-cudaxx.sh
1. wsl-install-nvidia-container-toolkit.sh
1. wsl-install-docker.sh


## CUDA operation check
Execute the following command and pass the test.
```
cd /usr/local/cuda/samples/4_Finance/BlackScholes
sudo make
./BlackScholes
```
The result is as follows.
```
:/usr/local/cuda/samples/4_Finance/BlackScholes# ./BlackScholes
[./BlackScholes] - Starting...
GPU Device 0: "Turing" with compute capability 7.5

Initializing data...
...allocating CPU memory for options.
...allocating GPU memory for options.
...generating input data in CPU mem.
...copying input data to GPU mem.
Data init done.

Executing Black-Scholes GPU kernel (512 iterations)...
Options count             : 8000000
BlackScholesGPU() time    : 0.273049 msec
Effective memory bandwidth: 292.987901 GB/s
Gigaoptions per second    : 29.298790

BlackScholes, Throughput = 29.2988 GOptions/s, Time = 0.00027 s, Size = 8000000 options, NumDevsUsed = 1, Workgroup = 128

Reading back GPU results...
Checking the results...
...running CPU calculations.

Comparing the results...
L1 norm: 1.741792E-07
Max absolute error: 1.192093E-05

Shutting down...
...releasing GPU memory.
...releasing CPU memory.
Shutdown done.

[BlackScholes] - Test Summary

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```