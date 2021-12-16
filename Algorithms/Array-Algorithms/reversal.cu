#include <device_launch_parameters.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void ArrayReversalKernel(T *array, unsigned arraySize) {
    unsigned bufferIdx = threadIdx.x;
    unsigned offset = blockDim.x * gridDim.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ T buffer[];
    for (unsigned i = idx; i < arraySize; i += offset) {
        array[i] = i + 1;
    }
    buffer[bufferIdx] = array[bufferIdx];
    __syncthreads();
    array[bufferIdx] = buffer[arraySize - bufferIdx - 1];
}