template <typename T>
__global__ void ArrayReductionKernel(T *inputArray, T *outputArray, unsigned arraySize) {
	extern __shared__ T intermediateSums[];
    unsigned threadIndex = threadIdx.x;
	unsigned globalIndex = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	intermediateSums[threadIndex] = 0;
	if (globalIndex < arraySize) {
		intermediateSums[threadIndex] = inputArray[globalIndex] + inputArray[globalIndex + blockDim.x];
	}
	__syncthreads();
	for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
		if (threadIndex < s) {
			intermediateSums[threadIndex] += intermediateSums[threadIndex + s];
		}
		__syncthreads();
	}
	// Assuming blockDim.x > 64
	if (threadIndex < 32) {
		intermediateSums[threadIndex] += intermediateSums[threadIndex + 32];
		intermediateSums[threadIndex] += intermediateSums[threadIndex + 16];
		intermediateSums[threadIndex] += intermediateSums[threadIndex + 8];
		intermediateSums[threadIndex] += intermediateSums[threadIndex + 4];
		intermediateSums[threadIndex] += intermediateSums[threadIndex + 2];
		intermediateSums[threadIndex] += intermediateSums[threadIndex + 1];
	}
	if (threadIndex == 0) {
		outputArray[blockIdx.x] = intermediateSums[0];
    }
}