#include <iostream>

// Define execution configuration variables
constexpr int numBlocksPerGrid = 1;
constexpr int numThreadsPerBlock = 64;

// Define hello world kernel
__global__ void HelloWorldKernel() {
    printf("Hello World № %d!\n", threadIdx.x * gridDim.x);
}

int main() {
    // HOST EXECUTION

    // Declare host clock variables
	float elapsedTimeHost;
    clock_t startTimeHost, stopTimeHost;

    // Start host clock
    startTimeHost = clock();

    // Launch execution on host
    for (int i = 0; i < numThreadsPerBlock; ++i) {
        std::cout << "Hello World №" << i << "!\n";
    }

    // Stop host clock
    stopTimeHost = clock();
    elapsedTimeHost = (float) ((stopTimeHost) - (startTimeHost));
	printf("Host Elapsed Time: %f ms\n", elapsedTimeHost);
    
    // DEVICE EXECUTION

    // Declare device clock variables
	float elapsedTimeDevice;
    cudaEvent_t startTimeDevice, stopTimeDevice;

    // Start device clock
    cudaEventCreate(&startTimeDevice);
	cudaEventRecord(startTimeDevice, 0);

    // Launch hello world kernel on device
    HelloWorldNonSharedMemoryKernel <<<numBlocksPerGrid, numThreadsPerBlock>>> ();

    // Wait for the device to finish computing
    cudaDeviceSynchronize();

    // Stop device clock
    cudaEventCreate(&stopTimeDevice);
	cudaEventRecord(stopTimeDevice, 0);
	cudaEventSynchronize(stopTimeDevice);
	cudaEventElapsedTime(&elapsedTimeDevice, startTimeDevice, stopTimeDevice);
    printf("Device Elapsed Time: %f ms\n", elapsedTimeDevice);

    // Check for any errors
    int exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }
    
    return exitStatus;
}
