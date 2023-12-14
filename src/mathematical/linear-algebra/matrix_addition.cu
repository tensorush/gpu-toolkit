#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>

// Define global constants
constexpr unsigned NUM_ROWS = 1 << 8;
constexpr unsigned NUM_COLS = 1 << 9;
constexpr unsigned BLOCK_DIM = 1 << 5;
constexpr unsigned NUM_STREAMS = 1 << 1;
constexpr unsigned TOTAL_SIZE = NUM_ROWS * NUM_COLS;
constexpr unsigned STREAM_SIZE = TOTAL_SIZE / NUM_STREAMS;
constexpr unsigned TOTAL_BYTES = TOTAL_SIZE * sizeof(float);
constexpr unsigned STREAM_BYTES = TOTAL_BYTES / NUM_STREAMS;

// Define matrix addition kernel
__global__ void MatrixAdditionKernel(float *A, float *B, float *C, unsigned streamIdx) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx = row * NUM_COLS + col;
    if (row < NUM_ROWS && col < NUM_COLS &&
        (streamIdx == 0 && idx <= STREAM_SIZE ||
        streamIdx == 1 && idx > STREAM_SIZE)) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Declare pointers to input and output data on host
    float *hostA = nullptr, *hostB = nullptr, *hostC = nullptr;

    // Allocate pinned host memory for input data
    cudaMallocHost(&hostA, TOTAL_BYTES);
    cudaMallocHost(&hostB, TOTAL_BYTES);
    cudaMallocHost(&hostC, TOTAL_BYTES);

    // Initialize input data on host
    for (unsigned row = 0; row < NUM_ROWS; ++row) {
        for (unsigned col = 0; col < NUM_COLS; ++col) {
            hostA[row * NUM_COLS + col] = 2.0f;
        }
    }
    for (unsigned row = 0; row < NUM_ROWS; ++row) {
        for (unsigned col = 0; col < NUM_COLS; ++col) {
            hostB[row * NUM_COLS + col] = 3.0f;
        }
    }
    
    // Declare pointers to input and output data on device
    float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;

    // Allocate device memory for input and output data
    cudaMalloc(&deviceA, TOTAL_BYTES);
    cudaMalloc(&deviceB, TOTAL_BYTES);
    cudaMalloc(&deviceC, TOTAL_BYTES);

    // Declare streams
    cudaStream_t streams[NUM_STREAMS];

    // Create streams
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Copy input data from host to device
    cudaMemcpyAsync(deviceA, hostA, STREAM_BYTES, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(deviceB, hostB, STREAM_BYTES, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(deviceA + STREAM_SIZE, hostA + STREAM_SIZE, STREAM_BYTES, cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpyAsync(deviceB + STREAM_SIZE, hostB + STREAM_SIZE, STREAM_BYTES, cudaMemcpyHostToDevice, streams[1]);

    // Declare event variables to measure execution time
    float elapsedTime_1, elapsedTime_2;
    cudaEvent_t startTime_1, startTime_2, endTime_1, endTime_2;

    // Create events to measure execution time
    cudaEventCreate(&startTime_1);
    cudaEventCreate(&startTime_2);
    cudaEventCreate(&endTime_1);
    cudaEventCreate(&endTime_2);

    // Define kernel configuration variables
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((NUM_ROWS - 1) / blockDim.x + 1, (NUM_COLS - 1) / blockDim.y + 1);

    // Launch matrix addition kernels on device and record start of execution
    MatrixAdditionKernel<<<gridDim, blockDim, 0, streams[0]>>>(deviceA, deviceB, deviceC, 0);
    cudaEventRecord(startTime_1, streams[0]);
    MatrixAdditionKernel<<<gridDim, blockDim, 0, streams[1]>>>(deviceA, deviceB, deviceC, 1);
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

    // Transfer output data from device to host
    cudaMemcpyAsync(hostC, deviceC, STREAM_BYTES, cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(hostC + STREAM_SIZE, deviceC + STREAM_SIZE, STREAM_BYTES, cudaMemcpyDeviceToHost, streams[1]);

    // Destroy streams
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    // Print output data on host
    std::cout << "C = A + B:\n";
    for (unsigned row = 0; row < NUM_ROWS; ++row) {
        for (unsigned col = 0; col < NUM_COLS; ++col) {
            std::cout << hostC[row * NUM_COLS + col] << ' ';
        }
        std::cout << '\n';
    }

    // Free device memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // Free pinned host memory
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);

    // Check for errors
    unsigned exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }

    return exitStatus;
}