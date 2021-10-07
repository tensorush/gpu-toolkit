// TODO: Finish complete program

__global__ void MatrixTranspose(float *odata, const float *idata) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}