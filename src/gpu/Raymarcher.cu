#include "gpu/Raymarcher.h"

#include "core/TransferFunction.h"
#include "gpu/CudaCheck.h"

#include <stdexcept>

namespace vs::gpu {
namespace {

struct KernelParams {
    float3 eye, forward, right, up;
    float3 boxMin, boxMax, background;
    float tanHalfFov, aspect, stepSize, densityScale, brightness;
    int maxSteps;
};

__host__ __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ inline float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }

__device__ inline float3 normalize3(float3 v) {
    return v * rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline bool intersectBox(float3 origin, float3 dir, float3 boxMin, float3 boxMax, float& tNear,
                                    float& tFar) {
    const float3 inv = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    const float3 t0 = (boxMin - origin) * inv;
    const float3 t1 = (boxMax - origin) * inv;
    tNear = fmaxf(fmaxf(fminf(t0.x, t1.x), fminf(t0.y, t1.y)), fminf(t0.z, t1.z));
    tFar = fminf(fminf(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y)), fmaxf(t0.z, t1.z));
    return tFar > fmaxf(tNear, 0.0f);
}

__device__ inline std::uint32_t packRgba(float3 c) {
    const auto channel = [](float v) { return static_cast<std::uint32_t>(__saturatef(v) * 255.0f + 0.5f); };
    return channel(c.x) | (channel(c.y) << 8) | (channel(c.z) << 16) | (255u << 24);
}

__global__ void raymarchKernel(cudaTextureObject_t volume, cudaTextureObject_t transfer, KernelParams p,
                               std::uint32_t* image, int width, int height) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) {
        return;
    }

    const float u = ((px + 0.5f) / width * 2.0f - 1.0f) * p.tanHalfFov * p.aspect;
    const float v = (1.0f - (py + 0.5f) / height * 2.0f) * p.tanHalfFov;
    const float3 dir = normalize3(p.forward + p.right * u + p.up * v);

    float tNear = 0.0f;
    float tFar = 0.0f;
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float alpha = 0.0f;

    if (intersectBox(p.eye, dir, p.boxMin, p.boxMax, tNear, tFar)) {
        const float3 extent = p.boxMax - p.boxMin;
        const float3 invExtent = make_float3(1.0f / extent.x, 1.0f / extent.y, 1.0f / extent.z);
        // Sample transfer-function texel centres: density 0 -> texel 0, density 1 -> last texel.
        constexpr float kTableSize = static_cast<float>(TransferFunction::kTableSize);
        constexpr float tfScale = (kTableSize - 1.0f) / kTableSize;
        constexpr float tfOffset = 0.5f / kTableSize;

        float t = fmaxf(tNear, 0.0f);
        for (int i = 0; i < p.maxSteps && t < tFar; ++i, t += p.stepSize) {
            const float3 uvw = (p.eye + dir * t - p.boxMin) * invExtent;
            const float density = tex3D<float>(volume, uvw.x, uvw.y, uvw.z);
            const float4 sample = tex1D<float4>(transfer, density * tfScale + tfOffset);

            // Opacity correction keeps the look stable when the step size changes.
            const float a = 1.0f - __expf(-sample.w * p.densityScale * p.stepSize);
            const float weight = (1.0f - alpha) * a;
            color = color + make_float3(sample.x, sample.y, sample.z) * weight;
            alpha += weight;
            if (alpha > 0.99f) {
                break; // Early ray termination.
            }
        }
    }

    const float3 result = color * p.brightness + p.background * (1.0f - alpha);
    image[py * width + px] = packRgba(result);
}

float3 toFloat3(Vec3 v) {
    return make_float3(v.x, v.y, v.z);
}

} // namespace

void Raymarcher::resize(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Render target size must be positive");
    }
    if (width == width_ && height == height_) {
        return;
    }
    image_ = DeviceBuffer<std::uint32_t>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    width_ = width;
    height_ = height;
}

float Raymarcher::render(cudaTextureObject_t volume, cudaTextureObject_t transfer, const RenderParams& params,
                         std::span<std::uint32_t> pixels) {
    if (image_.size() == 0) {
        throw std::logic_error("Raymarcher::resize must be called before render");
    }
    if (pixels.size() != image_.size()) {
        throw std::length_error("Pixel buffer size does not match the render target");
    }

    const KernelParams kernelParams{toFloat3(params.eye),     toFloat3(params.forward), toFloat3(params.right),
                                    toFloat3(params.up),      toFloat3(params.boxMin),  toFloat3(params.boxMax),
                                    toFloat3(params.background), params.tanHalfFov,    params.aspect,
                                    params.stepSize,          params.densityScale,      params.brightness,
                                    params.maxSteps};

    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned>(width_) + block.x - 1) / block.x,
                    (static_cast<unsigned>(height_) + block.y - 1) / block.y);

    GpuTimer timer;
    timer.start();
    raymarchKernel<<<grid, block>>>(volume, transfer, kernelParams, image_.get(), width_, height_);
    VS_CUDA_CHECK(cudaGetLastError());
    const float ms = timer.stop();

    image_.download(pixels);
    return ms;
}

} // namespace vs::gpu
