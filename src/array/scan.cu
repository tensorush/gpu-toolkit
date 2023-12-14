#include <device_launch_parameters.h>
#include <cuda_runtime.h>

// Define global constants in host memory
constexpr unsigned LOG_NUM_BANKS = 5;
constexpr unsigned MAX_BLOCK_SIZE = 1 << 7;
constexpr unsigned NUM_BANKS = 1 << LOG_NUM_BANKS;
constexpr unsigned BLOCK_SIZE = MAX_BLOCK_SIZE / 2;

// Determine offset to avoid memory bank conflicts
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) (n >> NUM_BANKS + n >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) (n >> LOG_NUM_BANKS)
#endif

// Define block sums scan kernel
__global__ void BlockSumsScanKernel(unsigned *output, const unsigned *input, unsigned *blockSums, const unsigned numElements) {
    unsigned blockSum = blockSums[blockIdx.x];
    unsigned idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = input[idx] + blockSum;
        idx += blockDim.x;
        if (idx < numElements) {
            output[idx] = input[idx] + blockSum;
        }
    }
}

// Define block scan kernel
__global__ void BlockScanKernel(unsigned *output, const unsigned *input, unsigned *blockSums, const unsigned gridSize) {
    // Conflict-free padding requires shared memory to be more than 2 * BLOCK_SIZE
    __shared__ unsigned blockSums[MAX_BLOCK_SIZE + (MAX_BLOCK_SIZE >> LOG_NUM_BANKS)];
    int cur = threadIdx.x;
    int next = cur + blockDim.x;
    // Zero out shared memory
    blockSums[cur] = 0;
    blockSums[next] = 0;
    // If CONFLICT_FREE_OFFSET is used, shared memory size must be 2 * blockDim.x + blockDim.x / NUM_BANKS
    blockSums[threadIdx.x + blockDim.x + (blockDim.x >> LOG_NUM_BANKS)] = 0;
    __syncthreads();
    // Copy input to shared memory
    unsigned idx = MAX_BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (idx < gridSize) {
        blockSums[cur + CONFLICT_FREE_OFFSET(cur)] = input[idx];
        if (idx + blockDim.x < gridSize) {
            blockSums[next + CONFLICT_FREE_OFFSET(next)] = input[idx + blockDim.x];
        }
    }
    // Reduction step
    int offset = 1;
    for (int blockSize = MAX_BLOCK_SIZE >> 1; blockSize > 0; blockSize >>= 1) {
        __syncthreads();
        if (threadIdx.x < blockSize) {
            int cur = offset * ((threadIdx.x << 1) + 1) - 1;
            int next = offset * ((threadIdx.x << 1) + 2) - 1;
            cur += CONFLICT_FREE_OFFSET(cur);
            next += CONFLICT_FREE_OFFSET(next);
            blockSums[next] += blockSums[cur];
        }
        offset <<= 1;
    }
    // Save current block sum and zero out last block sum
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = blockSums[MAX_BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(MAX_BLOCK_SIZE - 1)];
        blockSums[MAX_BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(MAX_BLOCK_SIZE - 1)] = 0;
    }
    for (int blockSize = 1; blockSize < MAX_BLOCK_SIZE; blockSize <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (threadIdx.x < blockSize) {
            int cur = offset * ((threadIdx.x << 1) + 1) - 1;
            int next = offset * ((threadIdx.x << 1) + 2) - 1;
            cur += CONFLICT_FREE_OFFSET(cur);
            next += CONFLICT_FREE_OFFSET(next);
            unsigned prev = blockSums[cur];
            blockSums[cur] = blockSums[next];
            blockSums[next] += prev;
        }
    }
    __syncthreads();
    if (idx < gridSize) {
        output[idx] = blockSums[cur + CONFLICT_FREE_OFFSET(cur)];
        idx += blockDim.x;
        if (idx < gridSize) {
            output[idx] = blockSums[next + CONFLICT_FREE_OFFSET(next)];
        }
    }
}

// Define scan function
void Scan(unsigned *output, const unsigned *input, const unsigned numElements) {
    // Zero out output
    cudaMemset(output, 0, numElements * sizeof(unsigned));

    // Determine grid size
    unsigned gridSize = (numElements - 1) / MAX_BLOCK_SIZE + 1;
    unsigned gridBytes = grifSize * sizeof(unsigned);

    // Allocate memory for block sums
    unsigned* blockSums = nullptr;
    cudaMalloc(&blockSums, gridBytes);
    cudaMemset(blockSums, 0, gridBytes);

    // Launch block scan kernel
    BlockScanKernel<<<gridSize, BLOCK_SIZE>>>(output, input, blockSums, numElements);

    // Scan block sums or recurse to get full block sums scan
    if (gridSize <= MAX_BLOCK_SIZE) {
        unsigned *dummyBlockSums = nullptr;
        cudaMalloc(&dummyBlockSums, sizeof(unsigned));
        cudaMemset(dummyBlockSums, 0, sizeof(unsigned));
        BlockScanKernel<<<1, BLOCK_SIZE>>>(blockSums, blockSums, dummyBlockSums, gridSize);
        cudaFree(dummyBlockSums);
    } else {
        unsigned *inputBlockSums = nullptr;
        cudaMalloc(&inputBlockSums, gridBytes);
        cudaMemcpy(inputBlockSums, blockSums, gridBytes, cudaMemcpyDeviceToDevice);
        Scan(blockSums, inputBlockSums, gridSize);
        cudaFree(inputBlockSums);
    }
    BlockSumsScanKernel<<<gridSize, BLOCK_SIZE>>>(output, output, blockSums, numElements);
    cudaFree(blockSums);
}