#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>

// Define global constants
constexpr unsigned BLOCK_DIM = 1 << 5;
constexpr unsigned MATRIX_DIM = 1 << 10;
constexpr unsigned TOTAL_SIZE = MATRIX_DIM * MATRIX_DIM;
constexpr unsigned TOTAL_BYTES = TOTAL_SIZE * sizeof(float);

// Define square matrix multiplication kernel
__global__ void SquareMatrixMultiplicationKernel(float *A, float *B, float *C) {
    __shared__ float tileA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float tileB[BLOCK_DIM][BLOCK_DIM];
    unsigned row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    unsigned col = blockIdx.x * BLOCK_DIM + threadIdx.x;
    float result = 0.0f;
    for (unsigned idx, blockIndex = 0; blockIndex < gridDim.x; ++blockIndex) {
        idx = row * MATRIX_DIM + blockIndex * BLOCK_DIM + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (idx < TOTAL_SIZE) ? (A[idx]) : (0);
        idx = (blockIndex * BLOCK_DIM + threadIdx.y) * MATRIX_DIM + col;
        tileB[threadIdx.y][threadIdx.x] = (idx < TOTAL_SIZE) ? (B[idx]) : (0);
        __syncthreads();
        for (unsigned k = 0; k < BLOCK_DIM; ++k) {
            result += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < MATRIX_DIM && col < MATRIX_DIM) {
        C[row * MATRIX_DIM + col] = result;
    }
}

int main() {
    // Declare pointers to input and output data on host
    float *hostA = nullptr, *hostB = nullptr, *hostC = nullptr;

    // Allocate pinned host memory for input data
    cudaMallocHost((void **) &hostA, TOTAL_BYTES);
    cudaMallocHost((void **) &hostB, TOTAL_BYTES);
    cudaMallocHost((void **) &hostC, TOTAL_BYTES);

    // Initialize input data on host
    for (unsigned row = 0; row < MATRIX_DIM; ++row) {
        for (unsigned col = 0; col < MATRIX_DIM; ++col) {
            hostA[row * MATRIX_DIM + col] = 2.0f;
        }
    }
    for (unsigned row = 0; row < MATRIX_DIM; ++row) {
        for (unsigned col = 0; col < MATRIX_DIM; ++col) {
            hostB[row * MATRIX_DIM + col] = 3.0f;
        }
    }
    
    // Declare pointers to input and output data on device
    float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;

    // Allocate device memory for input and output data
    cudaMalloc((void **) &deviceA, TOTAL_BYTES);
    cudaMalloc((void **) &deviceB, TOTAL_BYTES);
    cudaMalloc((void **) &deviceC, TOTAL_BYTES);

    // Copy input data from host to device
    cudaMemcpy(deviceA, hostA, TOTAL_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, TOTAL_BYTES, cudaMemcpyHostToDevice);

    // Declare event variables to measure execution time
    float elapsedTime;
    cudaEvent_t startTime, endTime;

    // Create events to measure execution time
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);

    // Define kernel configuration variables
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((MATRIX_DIM - 1) / blockDim.x + 1, (MATRIX_DIM - 1) / blockDim.y + 1);

    // Launch square matrix multiplication kernels on device
    SquareMatrixMultiplicationKernel<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC);

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
    std::cout << "Elapsed Time on Device: " << elapsedTime << " ms\n";

    // Destroy events
    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);

    // Transfer output data from device to host
    cudaMemcpy(hostC, deviceC, TOTAL_BYTES, cudaMemcpyDeviceToHost);

    // Print output data on host
    std::cout << "C = A x B:\n";
    for (unsigned row = 0; row < MATRIX_DIM; ++row) {
        for (unsigned col = 0; col < MATRIX_DIM; ++col) {
            std::cout << hostC[row * MATRIX_DIM + col] << ' ';
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