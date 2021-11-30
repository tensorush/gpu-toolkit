#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>

// Define global constants
constexpr unsigned BLOCK_DIM = 1 << 5;
constexpr unsigned NUM_STREAMS = 1 << 1;
constexpr char *GREETING = "Hello World";
constexpr unsigned NUM_GREETINGS = BLOCK_DIM;
constexpr unsigned NUM_GREETINGS_PER_STREAM = NUM_GREETINGS / NUM_STREAMS;

// Define hello world kernel
__global__ void HelloWorldKernel(unsigned streamIdx) {
    unsigned idx = NUM_GREETINGS_PER_STREAM * streamIdx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_GREETINGS) {
        printf("%s №%d!\n", GREETING, idx);
    }
}

int main() {
    // Declare streams
    cudaStream_t streams[NUM_STREAMS];

    // Create streams
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Declare event variables to measure execution time
    float elapsedTime_1, elapsedTime_2;
    cudaEvent_t startTime_1, startTime_2, endTime_1, endTime_2;

    // Create events to measure execution time
    cudaEventCreate(&startTime_1);
    cudaEventCreate(&startTime_2);
    cudaEventCreate(&endTime_1);
    cudaEventCreate(&endTime_2);

    // Define kernel configuration variables
    dim3 gridDim(1);
    dim3 blockDim(BLOCK_DIM);
    
    // Launch hello world kernel on device and record start of execution
    HelloWorldKernel<<<gridDim, blockDim, 0, streams[0]>>>(0);
    cudaEventRecord(startTime_1, streams[0]);
    HelloWorldKernel<<<gridDim, blockDim, 0, streams[1]>>>(1);
    cudaEventRecord(startTime_2, streams[1]);

    // Synchronize start of execution calls
    cudaEventSynchronize(startTime_1);
    cudaEventSynchronize(startTime_2);

    // Record end of execution
    cudaEventRecord(endTime_1, streams[0]);
    cudaEventRecord(endTime_2, streams[1]);

    // Synchronize end of execution calls
    cudaEventSynchronize(endTime_1);
    cudaEventSynchronize(endTime_2);

    // Calculate and print elapsed time
    cudaEventElapsedTime(&elapsedTime_1, startTime_1, endTime_1);
    cudaEventElapsedTime(&elapsedTime_2, startTime_2, endTime_2);
    std::cout << "Elapsed Time on Device Stream №1: " << elapsedTime_1 << " ms\n";
    std::cout << "Elapsed Time on Device Stream №2: " << elapsedTime_2 << " ms\n";

    // Destroy events
    cudaEventDestroy(startTime_1);
    cudaEventDestroy(startTime_2);
    cudaEventDestroy(endTime_1);
    cudaEventDestroy(endTime_2);

    // Check for any errors
    unsigned exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }
    
    return exitStatus;
}