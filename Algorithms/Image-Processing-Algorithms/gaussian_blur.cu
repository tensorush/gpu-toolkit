#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void ImageChannelsSeparationKernel(const uchar4 *const inputImageRGBA, unsigned numRows, unsigned numCols, unsigned char *const red, unsigned char *const green, unsigned char *const blue) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < numRows && col < numCols) {
        unsigned idx = row * numCols + col;
        red[idx] = inputImageRGBA[idx].x;
        green[idx] = inputImageRGBA[idx].y;
        blue[idx] = inputImageRGBA[idx].z;
    }
}

__global__ void ImageChannelsCombinationKernel(uchar4 *const outputImageRGBA, const unsigned numRows, const unsigned numCols, const unsigned char *const red, const unsigned char *const green, const unsigned char *const blue) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < numRows && col < numCols) {
        unsigned idx = row * numCols + col;
        outputImageRGBA[idx] = make_uchar4(red[idx], green[idx], blue[idx], 255);
    }
}

__global__ void GaussianBlurKernel(const float *const filter, const int filterSide, const unsigned char *const inputImage, unsigned char *const outputImage, const int numRows, const int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < numRows && col < numCols) {
        float blurredPixel = 0.0f;
        for (int i = -filterSide / 2; i <= filterSide / 2; ++i) {
            for (int j = -filterSide / 2; j <= filterSide / 2; ++j) {
                int filterWidth = filterSide / 2 + j;
                int filterHeight = filterSide / 2 + i;
                int imageWidth = std::min(std::max(0, col + j), numCols - 1);
                int imageHeight = std::min(std::max(0, row + i), numRows - 1);
                blurredPixel += inputImage[imageHeight * numCols + imageWidth] * filter[filterHeight * filterSide + filterWidth];
            }
        }
        outputImage[row * numCols + col] = blurredPixel;
    }
}