#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cmath>

// Define global constants in host memory
constexpr unsigned BLOCK_SIZE = 1 << 7;
constexpr unsigned NUM_SPLITS = 1 << 2;

// Define local radix sort kernel
__global__ void LocalRadixSortKernel(unsigned *output, unsigned *input, unsigned *prefixSums, unsigned *blockSums, const unsigned shiftWidth, const unsigned numElements) {
    extern __shared__ unsigned sharedMemory[];
    unsigned *inputs = sharedMemory;
    unsigned numMaskOutputs = BLOCK_SIZE + 1;
    unsigned *maskOutputs = &inputs[BLOCK_SIZE];
    unsigned *scannedMaskOutputs = &maskOutputs[numMaskOutputs];
    unsigned *maskOutputSums = &scannedMaskOutputs[BLOCK_SIZE];
    unsigned *scannedMaskOutputSums = &maskOutputSums[NUM_SPLITS];
    unsigned idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    inputs[threadIdx.x] = (idx < numElements) ? (input[idx]) : (0);
    __syncthreads();
    unsigned inputElement = inputs[threadIdx.x];
    unsigned twoBits = (inputElement >> shiftWidth) & 3;
    for (unsigned splitIdx = 0; splitIdx < NUM_SPLITS; ++splitIdx) {
        // Zero out mask outputs
        maskOutputs[threadIdx.x] = 0;
        if (threadIdx.x == 0) {
            maskOutputs[numMaskOutputs - 1] = 0;
        }
        __syncthreads();
        // Determine mask output
        bool doTwoBitsEqualSplit = false;
        if (idx < numElements) {
            doTwoBitsEqualSplit = (twoBits == splitIdx);
            maskOutputs[threadIdx.x] = doTwoBitsEqualSplit;
        }
        __syncthreads();
        // Scan mask outputs
        int partnerMaskOutput = 0;
        unsigned maskOutputSum = 0;
        unsigned numSteps = std::log2(BLOCK_SIZE);
        for (unsigned stepIdx = 0; stepIdx < numSteps; ++stepIdx) {
            partnerMaskOutput = threadIdx.x - (1 << stepIdx);
            maskOutputSum = (partnerMaskOutput < 0) ? (maskOutputs[threadIdx.x]) : (maskOutputs[threadIdx.x] + maskOutputs[partnerMaskOutput]);
            __syncthreads();
            maskOutputs[threadIdx.x] = maskOutputSum;
            __syncthreads();
        }
        unsigned maskOutput = maskOutputs[threadIdx.x];
        __syncthreads();
        maskOutputs[threadIdx.x + 1] = maskOutput;
        __syncthreads();
        if (threadIdx.x == 0) {
            maskOutputs[0] = 0;
            unsigned blockSum = maskOutputs[numMaskOutputs - 1];
            maskOutputSums[splitIdx] = blockSum;
            blockSums[splitIdx * gridDim.x + blockIdx.x] = blockSum;
        }
        __syncthreads();
        if (doTwoBitsEqualSplit && idx < numElements) {
            scannedMaskOutputs[threadIdx.x] = maskOutputs[threadIdx.x];
        }
        __syncthreads();
    }
    // Naive scan mask output sums
    if (threadIdx.x == 0) {
        unsigned scannedMaskOutputSum = 0;
        for (unsigned splitIdx = 0; splitIdx < NUM_SPLITS; ++splitIdx) {
            scannedMaskOutputSums[splitIdx] = scannedMaskOutputSum;
            scannedMaskOutputSum += maskOutputSums[splitIdx];
        }
    }
    __syncthreads();
    if (idx < numElements) {
        // Compute new indices for block elements
        unsigned prefixSum = scannedMaskOutputs[threadIdx.x];
        unsigned newIndex = prefixSum + scannedMaskOutputSums[twoBits];
        __syncthreads();
        // Shuffle block elements
        inputs[newIndex] = inputElement;
        scannedMaskOutputs[newIndex] = prefixSum;
        __syncthreads();
        prefixSums[idx] = scannedMaskOutputs[threadIdx.x];
        output[idx] = inputs[threadIdx.x];
    }
}

__global__ void BlockwiseShuffleKernel(unsigned *output, unsigned *input, unsigned *blockSumsScan, unsigned *prefixSums, const unsigned shiftWidth, const unsigned numElements) {
    unsigned idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        unsigned inputElement = input[idx];
        unsigned finalIndex = blockSumsScan[3 & (inputElement >> shiftWidth) * gridDim.x + blockIdx.x] + prefixSums[idx];
        __syncthreads();
        output[finalIndex] = inputElement;
    }
}

void RadixSort(unsigned *output, unsigned *input, const unsigned numElements) {
    unsigned gridSize = (numElements - 1) / BLOCK_SIZE + 1;
    unsigned *prefixSums = nullptr;
    cudaMalloc(&prefixSums, numElements * sizeof(unsigned));
    cudaMemset(prefixSums, 0, numElements * sizeof(unsigned));
    unsigned *blockSums = nullptr;
    unsigned numBlockSums = NUM_SPLITS * gridSize;
    cudaMalloc(&blockSums, numBlockSums * sizeof(unsigned));
    cudaMemset(blockSums, 0, numBlockSums * sizeof(unsigned));
    unsigned *blockSumsScan = nullptr;
    cudaMalloc(&blockSumsScan, numBlockSums * sizeof(unsigned));
    cudaMemset(blockSumsScan, 0, numBlockSums * sizeof(unsigned));
    unsigned numInputs = BLOCK_SIZE;
    unsigned numMaskOutputs = BLOCK_SIZE + 1;
    unsigned numScannedMaskOutputs = BLOCK_SIZE;
    unsigned numMaskOutputSums = NUM_SPLITS;
    unsigned numScannedMaskOutputSums = NUM_SPLITS;
    unsigned sharedBytes = (numInputs + numMaskOutputs + numScannedMaskOutputs + numMaskOutputSums + numScannedMaskOutputSums) * sizeof(unsigned);
    // Perform blockwise radix sort
    for (unsigned shiftWidth = 0; shiftWidth <= 30; shiftWidth += 2) {
        LocalRadixSortKernel<<<gridSize, BLOCK_SIZE, sharedBytes>>>(output, input, prefixSums, blockSums, shiftWidth, numElements);
        // Scan block sums
        Scan(blockSumsScan, blockSums, numBlockSums);
        // Shuffle blockwise sorted array to final positions
        BlockwiseShuffleKernel<<<gridSize, BLOCK_SIZE>>>(input, output, blockSumsScan, prefixSums, shiftWidth, numElements);
    }
    cudaMemcpy(output, input, numElements * sizeof(unsigned), cudaMemcpyDeviceToDevice);
    cudaFree(blockSumsScan);
    cudaFree(blockSums);
    cudaFree(prefixSums);
}