%%cu
#include <iostream>

// Define host constants
constexpr unsigned NUM_ROWS = 1 << 10;
constexpr unsigned NUM_COLS = 1 << 10;
constexpr unsigned BLOCK_DIM = 1 << 5;
constexpr unsigned NUM_STREAMS = 1 << 1;
constexpr unsigned TOTAL_SIZE = NUM_ROWS * NUM_COLS;
constexpr unsigned STREAM_SIZE = TOTAL_SIZE / NUM_STREAMS;
constexpr unsigned TOTAL_PITCH = TOTAL_SIZE * sizeof(float);
constexpr unsigned STREAM_PITCH = TOTAL_PITCH / NUM_STREAMS;

// Define device constants
__constant__ unsigned DEVICE_NUM_ROWS = NUM_ROWS;
__constant__ unsigned DEVICE_NUM_COLS = NUM_COLS;
__constant__ unsigned DEVICE_STREAM_SIZE = STREAM_SIZE;

// Define matrix addition kernel
__global__ void MatrixAdditionKernel(float *A, float *B, float *C, unsigned streamIdx) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned k = i * DEVICE_NUM_COLS + j;
    if (i < DEVICE_NUM_ROWS && j < DEVICE_NUM_COLS &&
        (streamIdx == 0 && k <= DEVICE_STREAM_SIZE ||
        streamIdx == 1 && k > DEVICE_STREAM_SIZE)) {
        C[k] = A[k] + B[k];
    }
}

int main() {
    // Declare pointers to input and output data on host
	float *hostA = nullptr, *hostB = nullptr, *hostC = nullptr;

    // Allocate pinned host memory for input data
    cudaMallocHost((void **) &hostA, TOTAL_PITCH);
    cudaMallocHost((void **) &hostB, TOTAL_PITCH);
    cudaMallocHost((void **) &hostC, TOTAL_PITCH);

    // Initialize input data on host
    for (unsigned i = 0; i < NUM_ROWS; ++i) {
        for (unsigned j = 0; j < NUM_COLS; ++j) {
            hostA[i * NUM_COLS + j] = 2.0f;
        }
    }
    for (unsigned i = 0; i < NUM_ROWS; ++i) {
        for (unsigned j = 0; j < NUM_COLS; ++j) {
            hostB[i * NUM_COLS + j] = 3.0f;
        }
    }
    
    // Declare pointers to input and output data on device
    float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;

    // Allocate device memory for input and output data
    cudaMalloc((void **) &deviceA, TOTAL_PITCH);
    cudaMalloc((void **) &deviceB, TOTAL_PITCH);
    cudaMalloc((void **) &deviceC, TOTAL_PITCH);

    // Declare streams
    cudaStream_t streams[NUM_STREAMS];

    // Create streams
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Copy input data from host to device
    cudaMemcpyAsync(deviceA, hostA, STREAM_PITCH, cudaMemcpyHostToDevice, streams[0]);
	cudaMemcpyAsync(deviceB, hostB, STREAM_PITCH, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(deviceA + STREAM_SIZE, hostA + STREAM_SIZE, STREAM_PITCH, cudaMemcpyHostToDevice, streams[1]);
	cudaMemcpyAsync(deviceB + STREAM_SIZE, hostB + STREAM_SIZE, STREAM_PITCH, cudaMemcpyHostToDevice, streams[1]);

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
    cudaMemcpyAsync(hostC, deviceC, STREAM_PITCH, cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(hostC + STREAM_SIZE, deviceC + STREAM_SIZE, STREAM_PITCH, cudaMemcpyDeviceToHost, streams[1]);

    // Destroy streams
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    // Print output data on host
    std::cout << "C = A + B:\n";
    for (unsigned i = 0; i < NUM_ROWS; ++i) {
        for (unsigned j = 0; j < NUM_COLS; ++j) {
            std::cout << hostC[i * NUM_COLS + j] << ' ';
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