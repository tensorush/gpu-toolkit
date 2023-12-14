template <typename T>
__global__ void MatrixTransposeKernel(const T *A, T *transposeA) {
    unsigned tileDim = blockIdx.x;
    unsigned numTiles = blockIdx.y;
    __shared__ T tile[tileDim][tileDim + 1];
    unsigned tileX = blockIdx.x * tileDim + threadIdx.x;
    unsigned tileY = blockIdx.y * tileDim + threadIdx.y;
    unsigned width = gridDim.x * tileDim;
    for (unsigned idx = 0; idx < tileDim; idx += numTiles) {
        tile[threadIdx.y + idx][threadIdx.x] = A[(tileY + idx) * width + tileX];
    }
    __syncthreads();
    tileX = blockIdx.y * tileDim + threadIdx.x;
    tileY = blockIdx.x * tileDim + threadIdx.y;
    for (unsigned idx = 0; idx < tileDim; idx += numTiles) {
        transposeA[(tileY + idx) * width + tileX] = tile[threadIdx.x][threadIdx.y + idx];
    }
}