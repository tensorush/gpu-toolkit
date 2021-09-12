#include <iostream>
#include <cmath>

// Define global constants
constexpr int NUM_ROW_ELEMENTS = 1 << 5;
constexpr int NUM_COL_ELEMENTS = 1 << 5;
constexpr int TOTAL_SIZE = NUM_ROW_ELEMENTS * NUM_COL_ELEMENTS * sizeof(int);

// Define data input kernel
template <typename T>
__global__ void DataInputKernel(T *d_A, T *d_B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NUM_ROW_ELEMENTS && j < NUM_COL_ELEMENTS) {
        d_A[i][j] = 2;
        d_B[i][j] = 3;
    }
}

// Define matrix addition kernel
template <typename T>
__global__ void MatrixAdditionKernel(T *d_A, T *d_B, T *d_C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < NUM_ROW_ELEMENTS && j < NUM_COL_ELEMENTS) {
        d_C[i][j] = d_A[i][j] + d_B[i][j];
    }
}

int main() {
    // Declare pointers to input data on device
    int(*d_A)[NUM_COL_ELEMENTS], (*d_B)[NUM_COL_ELEMENTS];

    // Declare pointers to output data on both host and device
    int(*h_C)[NUM_COL_ELEMENTS], (*d_C)[NUM_COL_ELEMENTS];

    // Allocate device memory
    cudaMalloc((void **) &d_A, TOTAL_SIZE);
    cudaMalloc((void **) &d_B, TOTAL_SIZE);
    cudaMalloc((void **) &d_C, TOTAL_SIZE);

    // Define execution configuration variables
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocksPerGrid(std::ceil(NUM_ROW_ELEMENTS / 32.0), std::ceil(NUM_COL_ELEMENTS / 32.0));

    // Launch data input kernel on device
    DataInputKernel << <numBlocksPerGrid, numThreadsPerBlock >> > (d_A, d_B);

    // Launch matrix addition kernel on device
    MatrixAdditionKernel << <numBlocksPerGrid, numThreadsPerBlock >> > (d_A, d_B, d_C);

    // Wait for the device to finish computing
    cudaDeviceSynchronize();

    // Transfer output data from device to host
    cudaMemcpy(h_C, d_C, TOTAL_SIZE, cudaMemcpyDeviceToHost);

    // Print output data
    printf("%d ", h_C);
    for (int i = 0; i < NUM_ROW_ELEMENTS; ++i) {
        for (int j = 0; j < NUM_COL_ELEMENTS; ++j) {
            std::cout << h_C[i][j] << ' ';
        }
    }
    std::cout << '\n';

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Check for errors
    int exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }

    return exitStatus;
}