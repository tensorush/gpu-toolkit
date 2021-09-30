#include <iostream>

// Define host constants
constexpr unsigned BLOCK_DIM = 1 << 5;
constexpr unsigned NUM_ROWS_A = 1 << 7;
constexpr unsigned SHARED_DIM = 1 << 8;
constexpr unsigned NUM_COLS_B = 1 << 9;
constexpr unsigned NUM_STREAMS = 1 << 1;
constexpr unsigned TOTAL_SIZE_A = NUM_ROWS_A * SHARED_DIM;
constexpr unsigned TOTAL_SIZE_B = SHARED_DIM * NUM_COLS_B;
constexpr unsigned TOTAL_SIZE_C = NUM_ROWS_A * NUM_COLS_B;
constexpr unsigned TOTAL_PITCH_A = TOTAL_SIZE_A * sizeof(float);
constexpr unsigned TOTAL_PITCH_B = TOTAL_SIZE_B * sizeof(float);
constexpr unsigned TOTAL_PITCH_C = TOTAL_SIZE_C * sizeof(float);

// Define device constants
__constant__ unsigned DEVICE_NUM_ROWS_A = NUM_ROWS_A;
__constant__ unsigned DEVICE_SHARED_DIM = SHARED_DIM;
__constant__ unsigned DEVICE_NUM_COLS_B = NUM_COLS_B;

// Define rectangular matrix multiplication kernel
__global__ void RectangularMatrixMultiplicationKernel(float *A, float *B, float *C) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    if (row < DEVICE_NUM_ROWS_A && col < DEVICE_NUM_COLS_B) {
        for (unsigned i = 0; i < DEVICE_SHARED_DIM; ++i) {
            result += A[row * DEVICE_SHARED_DIM + i] * B[i * DEVICE_NUM_COLS_B + col];
        }
        C[row * DEVICE_NUM_COLS_B + col] = result;
    }
}

int main() {
    // Declare pointers to input and output data on host
	float *hostA = nullptr, *hostB = nullptr, *hostC = nullptr;

    // Allocate pinned host memory for input data
    cudaMallocHost((void **) &hostA, TOTAL_PITCH_A);
    cudaMallocHost((void **) &hostB, TOTAL_PITCH_B);
    cudaMallocHost((void **) &hostC, TOTAL_PITCH_C);

    // Initialize input data on host
    for (unsigned i = 0; i < NUM_ROWS_A; ++i) {
        for (unsigned j = 0; j < SHARED_DIM; ++j) {
            hostA[i * SHARED_DIM + j] = 2.0f;
        }
    }
    for (unsigned i = 0; i < SHARED_DIM; ++i) {
        for (unsigned j = 0; j < NUM_COLS_B; ++j) {
            hostB[i * NUM_COLS_B + j] = 3.0f;
        }
    }
    
    // Declare pointers to input and output data on device
    float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;

    // Allocate device memory for input and output data
    cudaMalloc((void **) &deviceA, TOTAL_PITCH_A);
    cudaMalloc((void **) &deviceB, TOTAL_PITCH_B);
    cudaMalloc((void **) &deviceC, TOTAL_PITCH_C);

    // Copy input data from host to device
    cudaMemcpy(deviceA, hostA, TOTAL_PITCH_A, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, TOTAL_PITCH_B, cudaMemcpyHostToDevice);

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
    dim3 gridDim((NUM_ROWS_A - 1) / blockDim.x + 1, (NUM_COLS_B - 1) / blockDim.y + 1);

    // Launch rectangular matrix multiplication kernels on device and record start of execution
    RectangularMatrixMultiplicationKernel<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC);
    cudaEventRecord(startTime_1, 0);
    RectangularMatrixMultiplicationKernel<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC);
    cudaEventRecord(startTime_2, 0);
    
    // Synchronize start of execution calls
    cudaEventSynchronize(startTime_1);
    cudaEventSynchronize(startTime_2);

    // Record end of execution
    cudaEventRecord(endTime_1, 0);
    cudaEventRecord(endTime_2, 0);

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
    cudaMemcpy(hostC, deviceC, TOTAL_PITCH_C, cudaMemcpyDeviceToHost);

    // Print output data on host
    std::cout << "C = A x B:\n";
    for (unsigned i = 0; i < NUM_ROWS_A; ++i) {
        for (unsigned j = 0; j < NUM_COLS_B; ++j) {
            std::cout << hostC[i * NUM_COLS_B + j] << ' ';
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