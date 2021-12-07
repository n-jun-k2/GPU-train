#pragma once
#include <cuda_runtime.h>
#include <type_traits>

/*
  ステンシル計算：周囲の座標から新たな値を出力・更新を行う
*/

constexpr int CONSTANT_MEMORY_SIZE = 10;

/* 定数を保存するバッファを10個用意*/
__constant__ float cmem[CONSTANT_MEMORY_SIZE];

/**
 * @brief constat memory example kernel
 * f(x) ≒ c0(f(x + 4h) - f(x - 4h)) + c1(f(x + 3h) - f(x - 3h)) - c2(f(x + 2h) - f(x-2h)) + c3(f(x + h) -f(x - h))
 *
 * @tparam BDIM カーネルを起動するスレッド数=求めるステンシル数
 * @tparam RADIUS 値の計算に使用される点xの両側にある点の数を定義します。
 * @param in
 * @param out
 * @return Pointer of device memory wrapped with shared_ptr.
 */
template<int BDIM, int RADIUS, std::enable_if_t<RADIUS < CONSTANT_MEMORY_SIZE>* = nullptr>
__global__ void constant_example_kernel(float *in, float *out) {
  const int DIAMETER = 2 * RADIUS;

  __shared__ float smem[BDIM + DIAMETER];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int center_idx = threadIdx.x + RADIUS;

  smem[center_idx] = in[idx];

  if (threadIdx.x < RADIUS) {
    smem[threadIdx.x] = in[idx - RADIUS];
    smem[center_idx + BDIM] = in[idx + BDIM];
  }

  __syncthreads();

  float tmp = 0.0f;

  #pragma unroll
  for(int i = 1; i <= RADIUS; ++i) {
    tmp += cmem[i] * (smem[center_idx + i] - smem[center_idx - i]);
  }

  out[idx] = tmp;

}
