#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>

// Define global constants
constexpr unsigned BLOCK_DIM = 1 << 5;
constexpr unsigned NUM_ROWS_A = 1 << 7;
constexpr unsigned SHARED_DIM = 1 << 8;
constexpr unsigned NUM_COLS_B = 1 << 9;
constexpr unsigned TOTAL_SIZE_A = NUM_ROWS_A * SHARED_DIM;
constexpr unsigned TOTAL_SIZE_B = SHARED_DIM * NUM_COLS_B;
constexpr unsigned TOTAL_SIZE_C = NUM_ROWS_A * NUM_COLS_B;
constexpr unsigned TOTAL_BYTES_A = TOTAL_SIZE_A * sizeof(float);
constexpr unsigned TOTAL_BYTES_B = TOTAL_SIZE_B * sizeof(float);
constexpr unsigned TOTAL_BYTES_C = TOTAL_SIZE_C * sizeof(float);

// Define rectangular matrix multiplication kernel
__global__ void RectangularMatrixMultiplicationKernel(float *A, float *B, float *C) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    if (row < NUM_ROWS_A && col < NUM_COLS_B) {
        for (unsigned idx = 0; idx < SHARED_DIM; ++idx) {
            result += A[row * SHARED_DIM + idx] * B[idx * NUM_COLS_B + col];
        }
        C[row * NUM_COLS_B + col] = result;
    }
}

int main() {
    // Declare pointers to input and output data on host
    float *hostA = nullptr, *hostB = nullptr, *hostC = nullptr;

    // Allocate pinned host memory for input data
    cudaMallocHost((void **) &hostA, TOTAL_BYTES_A);
    cudaMallocHost((void **) &hostB, TOTAL_BYTES_B);
    cudaMallocHost((void **) &hostC, TOTAL_BYTES_C);

    // Assign input data on host
    for (unsigned row = 0; row < NUM_ROWS_A; ++row) {
        for (unsigned col = 0; col < SHARED_DIM; ++col) {
            hostA[row * SHARED_DIM + col] = 2.0f;
        }
    }
    for (unsigned row = 0; row < SHARED_DIM; ++row) {
        for (unsigned col = 0; col < NUM_COLS_B; ++col) {
            hostB[row * NUM_COLS_B + col] = 3.0f;
        }
    }
    
    // Declare pointers to input and output data on device
    float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;

    // Allocate device memory for input and output data
    cudaMalloc((void **) &deviceA, TOTAL_BYTES_A);
    cudaMalloc((void **) &deviceB, TOTAL_BYTES_B);
    cudaMalloc((void **) &deviceC, TOTAL_BYTES_C);

    // Copy input data from host to device
    cudaMemcpy(deviceA, hostA, TOTAL_BYTES_A, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, TOTAL_BYTES_B, cudaMemcpyHostToDevice);

    // Declare event variables to measure execution time
    float elapsedTime;
    cudaEvent_t startTime, endTime;

    // Create events to measure execution time
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);

    // Define kernel configuration variables
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((NUM_ROWS_A - 1) / blockDim.x + 1, (NUM_COLS_B - 1) / blockDim.y + 1);

    // Launch rectangular matrix multiplication kernels on device
    RectangularMatrixMultiplicationKernel<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC);

    // Record start of execution
    cudaEventRecord(startTime, 0);
    
    // Synchronize start of execution call
    cudaEventSynchronize(startTime);

    // Record end of execution
    cudaEventRecord(endTime, 0);

    // Synchronize end of execution call
    cudaEventSynchronize(endTime);

    // Calculate and print elapsed time
    cudaEventElapsedTime(&elapsedTime, startTime, endTime);
    std::cout << "Elapsed Time on Device Stream №1: " << elapsedTime << " ms\n";

    // Destroy events
    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);

    // Transfer output data from device to host
    cudaMemcpy(hostC, deviceC, TOTAL_BYTES_C, cudaMemcpyDeviceToHost);

    // Print output data on host
    std::cout << "C = A x B:\n";
    for (unsigned row = 0; row < NUM_ROWS_A; ++row) {
        for (unsigned col = 0; col < NUM_COLS_B; ++col) {
            std::cout << hostC[row * NUM_COLS_B + col] << ' ';
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