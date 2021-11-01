#include <iostream>
#include <cuda/std/atomic>

__global__ void kernel(unsigned int* const d_counter, cuda::std::atomic<unsigned int>* const d_atomic_counter) {
	const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

	printf("[%04lu] : counter = %u, atomic-counter = %u\n", tid, (*d_counter)++, (*d_atomic_counter)++);
}

int main() {
	unsigned int *d_counter;
	cuda::std::atomic<unsigned int> *d_atomic_counter;
	cudaMalloc(&d_counter, sizeof(unsigned int));
	cudaMalloc(&d_atomic_counter, sizeof(cuda::std::atomic<unsigned int>));

	kernel<<<4, 32>>>(d_counter, d_atomic_counter);
	cudaDeviceSynchronize();

	cudaFree(d_counter);
	cudaFree(d_atomic_counter);
}