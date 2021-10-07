%%cu
#include <iostream>

// Define global constants in host memory
constexpr unsigned NUM_DIGITS = 10;
constexpr unsigned MIN_NUMBER = 99;
constexpr unsigned MAX_NUMBER = 1000;
constexpr unsigned BLOCK_DIM = 1 << 5;
constexpr unsigned ARRAY_DIM = 1 << 10;
constexpr unsigned NUM_STREAMS = 1 << 1;
constexpr unsigned ARRAY_BYTES = ARRAY_DIM * sizeof(bool);
constexpr unsigned NUM_ELEMENTS_PER_STREAM = ARRAY_DIM / NUM_STREAMS;
constexpr unsigned CUBED_DIGITS_BYTES = NUM_DIGITS * sizeof(unsigned);
constexpr unsigned CUBED_DIGITS[NUM_DIGITS] = {0, 1, 8, 27, 64, 125, 216, 343, 512, 729};

// Define global array in device constant memory
__constant__ unsigned CUBED_DIGITS_DEVICE_CONSTANT[NUM_DIGITS] = {0, 1, 8, 27, 64, 125, 216, 343, 512, 729};

// Declare global array reference to device texture memory
texture<unsigned, 1, cudaReadModeElementType> CUBED_DIGITS_DEVICE_TEXTURE;

// Define three-digit Armstrong numbers calculation kernel
__global__ void ThreeDigitArmstrongNumbersKernel(bool *array, unsigned streamIdx) {
    unsigned number = NUM_ELEMENTS_PER_STREAM * streamIdx + blockIdx.x * blockDim.x + threadIdx.x;
    array[number] = false;
    if (number > MIN_NUMBER && number < MAX_NUMBER) {
        unsigned sumOfCubedDigits = 0;
        for (unsigned digit, digits = number; digits > 0; digits /= 10) {
            digit = digits % 10;
            // Read pre-computed data from constant device memory
            // sumOfCubedDigits += CUBED_DIGITS_DEVICE_CONSTANT[digit];
            // Read pre-computed data from texture device memory
            sumOfCubedDigits += tex1Dfetch(CUBED_DIGITS_DEVICE_TEXTURE, digit);
        }
        array[number] = (sumOfCubedDigits == number);
    }
}

int main() {
    std::cout << "HOST EXECUTION\n";

    // Declare host clock variables
    float elapsedTimeHost;
    clock_t startTimeHost, stopTimeHost;

    // Start host clock
    startTimeHost = clock();

    // Compute three-digit Armstrong numbers on host as on device
    std::cout << "Three-digit Armstrong numbers computed on host: ";
    unsigned sumOfCubedDigits;
    for (unsigned number = 0; number < ARRAY_DIM; ++number) {
        if (number > MIN_NUMBER && number < MAX_NUMBER) {
            sumOfCubedDigits = 0;
            for (unsigned digit, digits = number; digits > 0; digits /= 10) {
                digit = digits % 10;
                sumOfCubedDigits += CUBED_DIGITS[digit];
            }
            if (sumOfCubedDigits == number) {
                std::cout << number << ' ';
            }
        }
    }
    std::cout << '\n';

    // Stop host clock
    stopTimeHost = clock();
    elapsedTimeHost = stopTimeHost - startTimeHost;
    std::cout << "Elapsed Time on Host: " << elapsedTimeHost << " ms\n\n";

    std::cout << "DEVICE EXECUTION\n";

    // Declare array for output data on host
    bool hostArray[ARRAY_DIM];

    // Declare pointer to output data on device
    bool *deviceArray = nullptr;

    // Declare pointer to pre-computed data on device
    unsigned *deviceCubedDigits = nullptr;

    // Allocate device memory for output and pre-computed data
    cudaMalloc((void **) &deviceArray, ARRAY_BYTES);
    cudaMalloc((void **) &deviceCubedDigits, CUBED_DIGITS_BYTES);

    // Copy pre-computed data from host to device
    cudaMemcpy(deviceCubedDigits, CUBED_DIGITS, CUBED_DIGITS_BYTES, cudaMemcpyHostToDevice);

    // Bind pre-computed data to texture reference on device
    cudaBindTexture(0, CUBED_DIGITS_DEVICE_TEXTURE, deviceCubedDigits, CUBED_DIGITS_BYTES);

    // Declare streams
    cudaStream_t streams[NUM_STREAMS];

    // Create streams
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Declare event variables to measure execution time
    float elapsedTime_1, elapsedTime_2;
    cudaEvent_t startTime_1, startTime_2, endTime_1, endTime_2;

    // Create events to measure execution time
    cudaEventCreate(&startTime_1);
    cudaEventCreate(&startTime_2);
    cudaEventCreate(&endTime_1);
    cudaEventCreate(&endTime_2);

    // Define kernel configuration variables
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim((ARRAY_DIM - 1) / blockDim.x + 1);

    // Launch three-digit Armstrong numbers calculation kernel on device and record start of execution
    ThreeDigitArmstrongNumbersKernel<<<gridDim, blockDim, 0, streams[0]>>>(deviceArray, 0);
    cudaEventRecord(startTime_1, streams[0]);
    ThreeDigitArmstrongNumbersKernel<<<gridDim, blockDim, 0, streams[1]>>>(deviceArray, 1);
    cudaEventRecord(startTime_2, streams[1]);

    // Synchronize start of execution calls
    cudaEventSynchronize(startTime_1);
    cudaEventSynchronize(startTime_2);

    // Record end of execution
    cudaEventRecord(endTime_1, streams[0]);
    cudaEventRecord(endTime_2, streams[1]);

    // Synchronize end of execution calls
    cudaEventSynchronize(endTime_1);
    cudaEventSynchronize(endTime_2);

    // Calculate and print elapsed time
    cudaEventElapsedTime(&elapsedTime_1, startTime_1, endTime_1);
    cudaEventElapsedTime(&elapsedTime_2, startTime_2, endTime_2);
    std::cout << "Elapsed Time on Device Stream №1: " << elapsedTime_1 << " ms\n";
    std::cout << "Elapsed Time on Device Stream №2: " << elapsedTime_2 << " ms\n";

    // Destroy events
    cudaEventDestroy(startTime_1);
    cudaEventDestroy(startTime_2);
    cudaEventDestroy(endTime_1);
    cudaEventDestroy(endTime_2);

    // Transfer output data from device to host
    cudaMemcpy(hostArray, deviceArray, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Print output data on host
    std::cout << "Three-digit Armstrong numbers computed on device: ";
    for (unsigned number = 0; number < ARRAY_DIM; ++number) {
        if (hostArray[number]) {
            std::cout << number << ' ';
        }
    }
    std::cout << '\n';

    // Free device memory
    cudaFree(deviceArray);

    // Check for errors
    unsigned exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }

    return exitStatus;
}