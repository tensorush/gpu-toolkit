#include <iostream>

// Define hello world kernel
__global__ void HelloWorldKernel() {
    printf("Hello World № %d!\n", threadIdx.x * gridDim.x);
}

int main() {
    // Define execution configuration variables
    int numBlocksPerGrid = 1, numThreadsPerBlock = 32;

    // Launch hello world kernel on device
    HelloWorldKernel << <numBlocksPerGrid, numThreadsPerBlock >> > ();

    // Wait for the device to finish
    cudaDeviceSynchronize();

    // Check for errors
    int exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }

    return exitStatus;
}