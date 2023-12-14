#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Define global constants in host memory
constexpr unsigned BLOCK_DIM = 1 << 6;
constexpr long unsigned ARRAY_DIM = 1 << 6;
constexpr double goldenRatio = 1.618033988749895;
constexpr double squareRootOfFive = 2.23606797749979;
constexpr long unsigned ARRAY_BYTES = ARRAY_DIM * sizeof(long unsigned);

// Define Fibonacci numbers calculation kernel
__global__ void FibonacciNumbersKernel(long unsigned *fibonacciNumbers) {
	long unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	fibonacciNumbers[idx] = std::round(std::pow(goldenRatio, idx) / squareRootOfFive);
}

int main() {
    std::cout << "HOST KERNEL EXECUTION\n";

    // Declare array for output data on host
	long unsigned *hostFibonacciNumbers = nullptr;

    // Allocate host memory for output data
    cudaMallocHost(&hostFibonacciNumbers, ARRAY_BYTES);

    // Declare host clock variables
    float elapsedTimeHost;
    clock_t startTimeHost, stopTimeHost;

    // Start host clock
    startTimeHost = clock();

    // Compute Fibonacci numbers on host
    for (long unsigned i = 0; i < ARRAY_DIM; ++i) {
	    hostFibonacciNumbers[i] = std::round(std::pow(goldenRatio, i) / squareRootOfFive);
    }

    // Stop host clock
    stopTimeHost = clock();
    elapsedTimeHost = stopTimeHost - startTimeHost;
    std::cout << "Elapsed Time on Host: " << elapsedTimeHost << " ms\n";

    // Print output data on host
    std::cout << "Fibonacci numbers computed on host:\n";
	for (long unsigned i = 0; i < ARRAY_DIM; ++i) {
		std::cout << hostFibonacciNumbers[i] << ' ';
    }
	std::cout << '\n';

    std::cout << "\nDEVICE KERNEL EXECUTION\n";

    // Declare array for output data on device
	long unsigned *deviceFibonacciNumbers = nullptr;

    // Allocate device memory for output data
    cudaMalloc(&deviceFibonacciNumbers, ARRAY_BYTES);

    // Define kernel configuration variables
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim((ARRAY_DIM - 1) / blockDim.x + 1);

    // Declare event variables to measure execution time
    float elapsedTimeDevice;
    cudaEvent_t startTimeDevice, endTimeDevice;

    // Create events to measure execution time
    cudaEventCreate(&startTimeDevice);
    cudaEventCreate(&endTimeDevice);

	// Launch Fibonacci numbers calculation kernel on device
	FibonacciNumbersKernel<<<gridDim, blockDim>>>(deviceFibonacciNumbers);

    // Record start of execution
    cudaEventRecord(startTimeDevice, 0);
    
    // Synchronize start of execution call
    cudaEventSynchronize(startTimeDevice);

    // Record end of execution
    cudaEventRecord(endTimeDevice, 0);

    // Synchronize end of execution call
    cudaEventSynchronize(endTimeDevice);

    // Calculate and print elapsed time
    cudaEventElapsedTime(&elapsedTimeDevice, startTimeDevice, endTimeDevice);
    std::cout << "Elapsed Time on Device: " << elapsedTimeDevice << " ms\n";

    // Destroy events
    cudaEventDestroy(startTimeDevice);
    cudaEventDestroy(endTimeDevice);

    // Transfer output data from device to host
	cudaMemcpy(hostFibonacciNumbers, deviceFibonacciNumbers, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
    // Print output data on host
    std::cout << "Fibonacci numbers computed on device:\n";
	for (long unsigned i = 0; i < ARRAY_DIM; ++i) {
		std::cout << hostFibonacciNumbers[i] << ' ';
    }
	std::cout << '\n';

    // Free device memory
    cudaFree(deviceFibonacciNumbers);

    // Free host memory
    cudaFreeHost(hostFibonacciNumbers);

    // Check for errors
    unsigned exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }

	return exitStatus;
}