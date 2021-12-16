#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void RGBA2GreyKernel(const uchar4 *const rgbaImage, unsigned char *const greyImage, const unsigned numRows, const unsigned numCols) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned rowOffset = blockDim.x * gridDim.x;
    unsigned colOffset = blockDim.y * gridDim.y;
    for (j = col; j < numCols; j += colOffset) {
        for (i = row; i < numRows; i += rowOffset) {
            uchar4 rgba = rgbaImage[j * w + i];
            greyImage[j * w + i] = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
        }
    }
}
