#include "cuda_runtime.h"
#include <iostream>
#include <cmath>

const unsigned BLOCK_DIM = 1 << 5;
const unsigned ARRAY_DIM = 1 << 10;
const double goldenRatio = 1.618033988749895;
const double squareRootOfFive = 2.23606797749979;
const unsigned ARRAY_BYTES = ARRAY_DIM * sizeof(unsigned);

__global__ void FibonacciNumbersKernel(unsigned *fibonacciNumbers) {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	fibonacciNumbers[idx] = std::round(std::pow(goldenRatio, idx) / squareRootOfFive);
}

int main() {
    // Declare array for output data on host
	unsigned *hostFibonacciNumbers = nullptr;

    // Allocate host memory for output data
    cudaMallocHost((void **) &hostFibonacciNumbers, ARRAY_BYTES);

    // Declare array for output data on device
	unsigned *deviceFibonacciNumbers = nullptr;

    // Allocate device memory for output data
    cudaMalloc((void **) &deviceFibonacciNumbers, ARRAY_BYTES);

    // Define kernel configuration variables
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim((ARRAY_DIM - 1) / blockDim.x + 1);

	// Launch three-digit Armstrong numbers calculation kernel on device and record start of execution
	FibonacciNumbersKernel<<<gridDim, blockDim>>>(deviceFibonacciNumbers);

    // Transfer output data from device to host
	cudaMemcpy(hostFibonacciNumbers, deviceFibonacciNumbers, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
    // Print output data on host
    std::cout << "Fibonacci numbers computed on device:\n";
	for (unsigned i = 0; i < ARRAY_DIM; ++i) {
		std::cout << hostFibonacciNumbers[i] << ' ';
    }
	std::cout << '\n';

    // Check for errors
    unsigned exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }

	return exitStatus;
}