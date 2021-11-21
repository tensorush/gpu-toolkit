__global__ void DecolorizationKernel(const uchar4 *const rgbaImage, unsigned char *const greyImage, const unsigned numRows, const unsigned numCols) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx = row * numCols + col;
    if (row < numRows && col < numCols) {
        uchar4 rgba = rgbaImage[idx];
        float greyscale = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
        greyImage[idx] = greyscale;
    }
}