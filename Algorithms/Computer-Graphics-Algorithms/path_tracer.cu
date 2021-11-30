#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

constexpr float PI = 3.1415927f;
constexpr unsigned BLOCK_DIM = 8;
constexpr unsigned NUM_SPHERES = 9;
constexpr unsigned NUM_SAMPLES = 1024;
constexpr unsigned SCREEN_WIDTH = 512;
constexpr unsigned SCREEN_HEIGHT = 384;
constexpr unsigned NUM_RAY_BOUNCES = 3;
constexpr unsigned NUM_PIXELS = SCREEN_WIDTH * SCREEN_HEIGHT;
constexpr unsigned IMAGE_BYTES = NUM_PIXELS * sizeof(float3);

__device__ float3 Scale(const float3 &vector, float scalar) {
	return make_float3(scalar * vector.x, scalar * vector.y, scalar * vector.z);
}

__device__ float3 Add(const float3 &vector1, const float3 &vector2) {
	return make_float3(vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z);
}

__device__ float3 Subtract(const float3 &vector1, const float3 &vector2) {
	return make_float3(vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z);
}

__device__ float3 Multiply(const float3 &vector1, const float3 &vector2) {
	return make_float3(vector1.x * vector2.x, vector1.y * vector2.y, vector1.z * vector2.z);
}

__device__ float DotProduct(const float3 &vector1, const float3 &vector2) {
	return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z;
}

__device__ float3 Normalize(const float3 &vector) {
	return Scale(vector, 1.0f / sqrtf(DotProduct(vector, vector)));
}

__device__ float3 CrossProduct(const float3 &vector1, const float3 &vector2) {
	return make_float3(vector1.y * vector2.z - vector1.z * vector2.y, vector1.z * vector2.x - vector1.x * vector2.z, vector1.x * vector2.y - vector1.y * vector2.x);
}

__device__ float clampInRange(const float x, const float a, const float b) {
    return std::fmaxf(a, std::fminf(x, b));
}

__host__ float clampBetweenZeroAndOne(const float x) {
    return (x < 0.0f) ? (0.0f) : ((x > 1.0f) ? (1.0f) : (x));
}

__host__ int convertColourFromFloatToInt(const float x) {
    return int(std::pow(clampBetweenZeroAndOne(x), 1 / 2.2f) * 255.0f + 0.5f);
}

enum MaterialReflectionType {
    DIFFUSE
};

struct Ray { 
    float3 origin;
    float3 direction;
    __device__ Ray(float3 origin_, float3 direction_) : origin(origin_), direction(direction_) {}
};

struct Sphere {
    float radius;
    float3 center;
    float3 colour;
    float3 emission;
    MaterialReflectionType reflectionType;
    __device__ float computeHitDistanceFromRayOriginToSphere(const Ray &ray) const {
        /*
            Ray equation:
                p(x, y, z) = ray.origin + hitDistance * ray.direction
            Sphere equation:
                x^2 + y^2 + z^2 = radius^2
            Quadratic equation:
                ax^2 + bx + c = 0
            Quadratic equation solutions:
                x1, x2 = (-b +- sqrt(b^2 - 4ac)) / 2a
            Solve for hitDistance:
                hitDistance^2 * ray.direction^2 + 2t * (origin - p) * ray.direction + (origin - p)^2 - radius^2 = 0
        */
        // Declare ray-to-sphere hit distance
        float hitDistance;
        // Define epsilon to aid floating point imprecision
        float epsilon = 0.0001f;
        // Distance from ray origin to sphere center
        float3 raySphereDistance = Subtract(center, ray.origin);
        // Find b coefficient of quadratic equation
        float b = DotProduct(raySphereDistance, ray.direction);
        // Find discriminant of quadratic equation
        float discriminant = b * b - DotProduct(raySphereDistance, raySphereDistance) + radius * radius;
        // If discriminant < 0, no real solutions
        if (discriminant < 0) {
            return 0;
        }
        // If discriminant >= 0, find real solutions
        discriminant = sqrtf(discriminant);
        // Return shortest hit distance from ray origin to sphere
        return ((hitDistance = b - discriminant) > epsilon) ? (hitDistance) : (((hitDistance = b + discriminant) > epsilon) ? (hitDistance) : (0));
    }
};

// Scene: Cornell Box
__constant__ Sphere SPHERES[] = {
    { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.75f, 0.75f, 0.75f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }, //Back
    { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.75f, 0.75f, 0.75f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }, //Bottom
    { 16.5f, { 27.0f, 16.5f, 47.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }, // Small 1
    { 16.5f, { 73.0f, 16.5f, 78.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }, // Small 2
    { 600.0f, { 50.0f, 680.83f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 2.0f, 1.8f, 1.6f }, DIFFUSE }, // Light
    { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.75f, 0.25f, 0.25f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }, // Left
    { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }, // Front
    { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.75f, 0.75f, 0.75f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }, // Top
    { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.25f, 0.25f, 0.75f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }, // Right
};

__device__ bool DoesRayIntersectSphere(const Ray &ray, float &hitDistance, unsigned &hitSphereIndex) {
    hitDistance = INFINITY;
    float smallerHitDistance;
    // Check every sphere for ray hit
    for (unsigned sphereIndex = 0; sphereIndex < NUM_SPHERES; ++sphereIndex) {
        // Update hit distance and hit sphere index
        if ((smallerHitDistance = SPHERES[sphereIndex].computeHitDistanceFromRayOriginToSphere(ray)) && smallerHitDistance < hitDistance) {
            hitDistance = smallerHitDistance;
            hitSphereIndex = sphereIndex;
        }
    }
    return hitDistance < INFINITY;
}

__device__ float GenerateRandomNumber(const unsigned seed1, const unsigned seed2) {
    unsigned hash1 = 36'969 * (seed1 & 65'535) + (seed1 >> 16);
    unsigned hash2 = 18'000 * (seed2 & 65'535) + (seed2 >> 16);
    return (((((hash1 << 16) + hash2) & 8'388'607) | 1'073'741'824) - 2) / 2.0f;
}

__device__ float3 TraceRayPath(Ray &ray, const unsigned seed1, const unsigned seed2) {
    /*
        Rendering equation:
            Outgoing Radiance = Emitted Radiance + Reflected Radiance,
            where Reflected Radiance is an integral of incoming radiance over the hemisphere above the hit point
            multiplied by the bidirectional reflectence distribution function of the hit material (BRDF) and cosine of incident angle
    */
    // Define colour bleeding to be neutral
    float3 colourBleeding = make_float3(1.0f, 1.0f, 1.0f);
    // Define accumulated ray colour to be black
    float3 accumulatedRayColour = make_float3(0.0f, 0.0f, 0.0f);
    // Bounce ray around scene
    for (unsigned bounceIndex = 0; bounceIndex < NUM_RAY_BOUNCES; ++bounceIndex) {
        float hitDistance;
        unsigned hitSphereIndex;
        // Check every sphere for ray hit
        if (!DoesRayIntersectSphere(ray, hitDistance, hitSphereIndex)) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
        // Compute hit point and normal
        Sphere &hitSphere = SPHERES[hitSphereIndex];
        float3 hitPoint = Add(ray.origin, Scale(ray.direction, hitDistance));
        float3 hitNormal = Normalize(Subtract(hitPoint, hitSphere.center));
        if (DotProduct(hitNormal, ray.direction) >= 0) {
            hitNormal = Scale(hitNormal, -1.0f);
        }
        // Add emitted light to accumulated ray colour
        accumulatedRayColour = Add(accumulatedRayColour, Multiply(colourBleeding, hitSphere.emission));
        // Generate random azimuth and elevation level for new ray direction
        float azimuth = 2 * PI * GenerateRandomNumber(seed1, seed2);
        float elevation = sqrtf(GenerateRandomNumber(seed1, seed2));
        // Construct orthonormal basis to generate random ray direction
        float3 unitNormal1 = hitNormal;
        float3 someNormal = (std::abs(unitNormal1.x) > 0.1f) ? (make_float3(0.0f, 1.0f, 0.0f)) : (make_float3(1.0f, 0.0f, 0.0f));
        float3 unitNormal2 = Normalize(CrossProduct(someNormal, unitNormal1));
        float3 unitNormal3 = CrossProduct(unitNormal1, unitNormal2);
        // Generate random ray direction on hemisphere using polar coordinates and cosine weighted importance sampling, which favours ray directions closer to normal
        unitNormal1 = Scale(unitNormal1, sqrtf(1.0f - elevation));
        unitNormal2 = Scale(unitNormal2, std::cos(azimuth) * elevation);
        unitNormal3 = Scale(unitNormal3, std::sin(azimuth) * elevation);
        ray.direction = Normalize(Add(unitNormal1, Add(unitNormal2, unitNormal3)));
        // Offset ray origin slightly to prevent self intersection
        ray.origin = Add(hitPoint, Scale(hitNormal, 0.05f));
        // Weight light contribution by cosine of angle between outgoing light and normal
        colourBleeding = Scale(colourBleeding, DotProduct(ray.direction, hitNormal));
        // Weight by sphere colour
        colourBleeding = Multiply(colourBleeding, hitSphere.colour);
        // Weight by BRDF fudge factor
        colourBleeding = Scale(colourBleeding, 2.0f);
    }
    return accumulatedRayColour;
}

__global__ void PathTracingKernel(float3 *image) {
    // Assign each thread to pixel
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned curPixel = (SCREEN_HEIGHT - y - 1) * SCREEN_WIDTH + x;
    unsigned seed1 = x;
    unsigned seed2 = y;
    // Generate ray directed at lower left screen corner
    Ray ray(make_float3(50.0f, 52.0f, 295.6f), Normalize(make_float3(0.0f, -0.042612f, -1.0f)));
    // Compute directions for other rays by adding rayOffsetX in x and y directions
    float fieldOfViewAngle = 0.5135f;
    float3 pixelColour = make_float3(0.0f, 0.0f, 0.0f);
    float3 rayOffsetX = make_float3(SCREEN_WIDTH * fieldOfViewAngle / SCREEN_HEIGHT, 0.0f, 0.0f);
    float3 rayOffsetY = Scale(Normalize(CrossProduct(rayOffsetX, ray.direction)), fieldOfViewAngle);
    // Sample rays
    for (unsigned sample = 0; sample < NUM_SAMPLES; ++sample) {
        // Compute primary ray direction
        float3 direction = Add(ray.direction, Add(Scale(rayOffsetX, (x + 0.25f) / SCREEN_WIDTH - 0.5f), Scale(rayOffsetY, (y + 0.25f) / SCREEN_HEIGHT - 0.5f)));
        // Create primary ray
        Ray primaryRay = Ray(Add(ray.origin, Scale(direction, 40.0f)), Normalize(direction));
        // Add traced ray path to pixel color
        pixelColour = Add(pixelColour, Scale(TraceRayPath(primaryRay, seed1, seed2), 1.0f / NUM_SAMPLES));
    }
    // Clamp floating-point pixel colour in range [0; 1]
    image[curPixel] = make_float3(clampInRange(pixelColour.x, 0.0f, 1.0f), clampInRange(pixelColour.y, 0.0f, 1.0f), clampInRange(pixelColour.z, 0.0f, 1.0f));
}

int main() {
    // Declare pointer for output image on host
    float3 *hostImage = nullptr;

    // Allocate host memory for output image
    // cudaMallocHost((void **) &hostImage, IMAGE_BYTES);
    cudaMallocHost(&hostImage, IMAGE_BYTES);

    // Declare pointer for output image on device
    float3 *deviceImage = nullptr;

    // Allocate device memory for output image
    // cudaMalloc((void **) &deviceImage, IMAGE_BYTES);
    cudaMalloc(&deviceImage, IMAGE_BYTES);

    // Define kernel configuration variables
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(SCREEN_WIDTH / blockDim.x, SCREEN_HEIGHT / blockDim.y);

    // Launch render kernel on device
    PathTracingKernel<<<gridDim, blockDim>>>(deviceImage);

    // Copy output image from device to host
    cudaMemcpy(hostImage, deviceImage, IMAGE_BYTES, cudaMemcpyDeviceToHost);

    // Write output image to .ppm file
    FILE *imageFile = fopen("cuda_path_tracer.ppm", "w");
    fprintf(imageFile, "P3\n%d %d\n%d\n", SCREEN_WIDTH, SCREEN_HEIGHT, 255);
    for (unsigned i = 0; i < NUM_PIXELS; ++i) {
        fprintf(imageFile, "%d %d %d ", convertColourFromFloatToInt(hostImage[i].x), convertColourFromFloatToInt(hostImage[i].y), convertColourFromFloatToInt(hostImage[i].z));
    }

    // Free device memory
    cudaFree(deviceImage);

    // Free host memory
    cudaFreeHost(hostImage);
    
    // Check for errors
    unsigned exitStatus = EXIT_SUCCESS;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << '\n';
        exitStatus = EXIT_FAILURE;
    }

    return exitStatus;
}