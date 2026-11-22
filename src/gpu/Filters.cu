#include "gpu/Filters.h"

#include "core/CpuFilters.h"
#include "gpu/CudaCheck.h"

#include <stdexcept>
#include <vector>

namespace vs::gpu {
namespace {

__constant__ float c_weights[2 * kMaxGaussianRadius + 1];

// One thread per output voxel; Axis 0/1/2 = X/Y/Z.
// TODO(stage 2): stage the neighbourhood in shared memory and compare against this naive version.
template <int Axis>
__global__ void convolveAxis(const float* __restrict__ input, float* __restrict__ output, int nx, int ny, int nz,
                             int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        int sx = x;
        int sy = y;
        int sz = z;
        if constexpr (Axis == 0) {
            sx = min(max(x + k, 0), nx - 1);
        } else if constexpr (Axis == 1) {
            sy = min(max(y + k, 0), ny - 1);
        } else {
            sz = min(max(z + k, 0), nz - 1);
        }
        const std::size_t index = (static_cast<std::size_t>(sz) * ny + sy) * nx + sx;
        sum += c_weights[k + radius] * input[index];
    }
    output[(static_cast<std::size_t>(z) * ny + y) * nx + x] = sum;
}

} // namespace

float gaussianBlur(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, Dims dims, float sigma) {
    const std::size_t count = dims.voxelCount();
    if (!dims.valid() || input.size() != count || output.size() != count) {
        throw std::invalid_argument("gaussianBlur: buffer sizes do not match the volume dimensions");
    }
    if (input.get() == output.get()) {
        throw std::invalid_argument("gaussianBlur: input and output must not alias");
    }

    const std::vector<float> weights = makeGaussianKernel(sigma);
    const int radius = static_cast<int>(weights.size() / 2);
    VS_CUDA_CHECK(cudaMemcpyToSymbol(c_weights, weights.data(), weights.size() * sizeof(float)));

    DeviceBuffer<float> temp(count);
    const dim3 block(8, 8, 8);
    const dim3 grid((static_cast<unsigned>(dims.x) + block.x - 1) / block.x,
                    (static_cast<unsigned>(dims.y) + block.y - 1) / block.y,
                    (static_cast<unsigned>(dims.z) + block.z - 1) / block.z);

    GpuTimer timer;
    timer.start();
    convolveAxis<0><<<grid, block>>>(input.get(), output.get(), dims.x, dims.y, dims.z, radius);
    convolveAxis<1><<<grid, block>>>(output.get(), temp.get(), dims.x, dims.y, dims.z, radius);
    convolveAxis<2><<<grid, block>>>(temp.get(), output.get(), dims.x, dims.y, dims.z, radius);
    VS_CUDA_CHECK(cudaGetLastError());
    return timer.stop();
}

Volume gaussianBlurGpu(const Volume& input, float sigma, float* gpuMs) {
    const Dims dims = input.dims();
    DeviceBuffer<float> deviceInput(dims.voxelCount());
    DeviceBuffer<float> deviceOutput(dims.voxelCount());
    deviceInput.upload(input.voxels());

    const float ms = gaussianBlur(deviceInput, deviceOutput, dims, sigma);
    if (gpuMs != nullptr) {
        *gpuMs = ms;
    }

    std::vector<float> voxels(dims.voxelCount());
    deviceOutput.download(voxels);
    Volume result(dims, std::move(voxels), input.spacing());
    result.setName(input.name());
    return result;
}

} // namespace vs::gpu
