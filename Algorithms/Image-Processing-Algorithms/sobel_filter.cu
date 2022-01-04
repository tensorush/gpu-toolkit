#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cmath>

texture<int, 2> image;

__global__ void SobelFilterKernel(int *filteredImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (width > x && height > y) {
        int dx =    tex2D(image, x - 1, y - 1) +     -tex2D(image, x + 1, y - 1) +
                2 * tex2D(image, x - 1, y    ) + -2 * tex2D(image, x + 1, y    ) +
                    tex2D(image, x - 1, y + 1) +     -tex2D(image, x + 1, y + 1);
        int dy =    tex2D(image, x - 1, y - 1) +     -tex2D(image, x - 1, y + 1) +
                2 * tex2D(image, x    , y - 1) + -2 * tex2D(image, x    , y + 1) +
                    tex2D(image, x + 1, y - 1) +     -tex2D(image, x + 1, y + 1);
        filteredImage[x + y * gridDim.x * blockDim.x] = std::sqrt(static_cast<double>(dx * dx + dy * dy));
    }
}
