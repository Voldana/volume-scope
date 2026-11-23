#include "gpu/Filters.h"

#include "core/CpuFilters.h"
#include "gpu/CudaCheck.h"

#include <stdexcept>
#include <vector>

namespace vs::gpu {
namespace {

__constant__ float c_weights[2 * kMaxGaussianRadius + 1];

__device__ inline int clampi(int v, int lo, int hi) {
    return min(max(v, lo), hi);
}

__device__ inline std::size_t voxelIndex(int x, int y, int z, int nx, int ny) {
    return (static_cast<std::size_t>(z) * ny + y) * nx + x;
}

/// Clamp-to-edge read.
__device__ inline float load(const float* __restrict__ v, int x, int y, int z, int nx, int ny, int nz) {
    return v[voxelIndex(clampi(x, 0, nx - 1), clampi(y, 0, ny - 1), clampi(z, 0, nz - 1), nx, ny)];
}

// ---------------------------------------------------------------------------
// Gaussian, naive: one thread per output voxel; Axis 0/1/2 = X/Y/Z.
// ---------------------------------------------------------------------------
template <int Axis>
__global__ void convolveNaive(const float* __restrict__ input, float* __restrict__ output, int nx, int ny, int nz,
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
            sx = clampi(x + k, 0, nx - 1);
        } else if constexpr (Axis == 1) {
            sy = clampi(y + k, 0, ny - 1);
        } else {
            sz = clampi(z + k, 0, nz - 1);
        }
        sum += c_weights[k + radius] * input[voxelIndex(sx, sy, sz, nx, ny)];
    }
    output[voxelIndex(x, y, z, nx, ny)] = sum;
}

// ---------------------------------------------------------------------------
// Gaussian, tiled X pass: a block convolves a 128-voxel run of one row. The run plus its
// halo is loaded once into shared memory (coalesced), then every tap reads shared memory.
// ---------------------------------------------------------------------------
constexpr int kRowTile = 128;

__global__ void convolveTiledX(const float* __restrict__ input, float* __restrict__ output, int nx, int ny,
                               int radius) {
    __shared__ float tile[kRowTile + 2 * kMaxGaussianRadius];
    const int x0 = blockIdx.x * kRowTile;
    const int y = blockIdx.y;
    const int z = blockIdx.z;
    const std::size_t row = voxelIndex(0, y, z, nx, ny);

    for (int i = threadIdx.x; i < kRowTile + 2 * radius; i += kRowTile) {
        tile[i] = input[row + clampi(x0 + i - radius, 0, nx - 1)];
    }
    __syncthreads();

    const int x = x0 + threadIdx.x;
    if (x >= nx) {
        return;
    }
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        sum += c_weights[k + radius] * tile[threadIdx.x + radius + k];
    }
    output[row + x] = sum;
}

// ---------------------------------------------------------------------------
// Gaussian, tiled Y or Z pass. Threads stay contiguous in x so global loads coalesce; the
// block covers kColTile voxels along the convolution axis (each thread computes
// kOutputsPerThread of them) and stages them plus the halo in shared memory.
// ---------------------------------------------------------------------------
constexpr int kTileX = 32;
constexpr int kBlockRows = 8;
constexpr int kOutputsPerThread = 4;
constexpr int kColTile = kBlockRows * kOutputsPerThread;

template <int Axis> // 1 = Y, 2 = Z
__global__ void convolveTiledYZ(const float* __restrict__ input, float* __restrict__ output, int nx, int ny, int nz,
                                int radius) {
    __shared__ float tile[kColTile + 2 * kMaxGaussianRadius][kTileX + 1]; // +1 avoids bank conflicts.
    const int x = blockIdx.x * kTileX + threadIdx.x;
    const int a0 = blockIdx.y * kColTile;   // Start along the convolution axis.
    const int other = blockIdx.z;           // The remaining axis (z for a Y pass, y for a Z pass).
    const int axisLength = Axis == 1 ? ny : nz;

    const auto index = [&](int a) {
        return Axis == 1 ? voxelIndex(x, a, other, nx, ny) : voxelIndex(x, other, a, nx, ny);
    };

    if (x < nx) {
        for (int r = threadIdx.y; r < kColTile + 2 * radius; r += kBlockRows) {
            tile[r][threadIdx.x] = input[index(clampi(a0 + r - radius, 0, axisLength - 1))];
        }
    }
    __syncthreads();
    if (x >= nx) {
        return;
    }

    for (int i = 0; i < kOutputsPerThread; ++i) {
        const int localRow = threadIdx.y + i * kBlockRows;
        const int a = a0 + localRow;
        if (a >= axisLength) {
            break;
        }
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            sum += c_weights[k + radius] * tile[localRow + radius + k][threadIdx.x];
        }
        output[index(a)] = sum;
    }
}

// ---------------------------------------------------------------------------
// Median and Sobel: one thread per voxel.
// ---------------------------------------------------------------------------
__global__ void median3Kernel(const float* __restrict__ input, float* __restrict__ output, int nx, int ny, int nz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    float window[27];
    int n = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                window[n++] = load(input, x + dx, y + dy, z + dz, nx, ny, nz);
            }
        }
    }
    // Partial selection sort: after 14 rounds, window[13] is the median.
    for (int i = 0; i <= 13; ++i) {
        int smallest = i;
        for (int j = i + 1; j < 27; ++j) {
            if (window[j] < window[smallest]) {
                smallest = j;
            }
        }
        const float tmp = window[i];
        window[i] = window[smallest];
        window[smallest] = tmp;
    }
    output[voxelIndex(x, y, z, nx, ny)] = window[13];
}

__global__ void sobelKernel(const float* __restrict__ input, float* __restrict__ output, int nx, int ny, int nz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const float smooth[3] = {1.0f, 2.0f, 1.0f};
    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;
    for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
            const float w = smooth[a + 1] * smooth[b + 1];
            gx += w * (load(input, x + 1, y + a, z + b, nx, ny, nz) - load(input, x - 1, y + a, z + b, nx, ny, nz));
            gy += w * (load(input, x + a, y + 1, z + b, nx, ny, nz) - load(input, x + a, y - 1, z + b, nx, ny, nz));
            gz += w * (load(input, x + a, y + b, z + 1, nx, ny, nz) - load(input, x + a, y + b, z - 1, nx, ny, nz));
        }
    }
    constexpr float kNorm = 1.0f / 32.0f;
    gx *= kNorm;
    gy *= kNorm;
    gz *= kNorm;
    output[voxelIndex(x, y, z, nx, ny)] = fminf(sqrtf(gx * gx + gy * gy + gz * gz), 1.0f);
}

void validate(const DeviceBuffer<float>& input, const DeviceBuffer<float>& output, Dims dims) {
    const std::size_t count = dims.voxelCount();
    if (!dims.valid() || input.size() != count || output.size() != count) {
        throw std::invalid_argument("GPU filter: buffer sizes do not match the volume dimensions");
    }
    if (input.get() == output.get()) {
        throw std::invalid_argument("GPU filter: input and output must not alias");
    }
}

dim3 voxelGrid(Dims dims, dim3 block) {
    return dim3((static_cast<unsigned>(dims.x) + block.x - 1) / block.x,
                (static_cast<unsigned>(dims.y) + block.y - 1) / block.y,
                (static_cast<unsigned>(dims.z) + block.z - 1) / block.z);
}

unsigned divUp(int value, int divisor) {
    return static_cast<unsigned>((value + divisor - 1) / divisor);
}

/// Upload, run `filter(deviceInput, deviceOutput, dims)`, download.
Volume runOnGpu(const Volume& input, float* gpuMs, const auto& filter) {
    const Dims dims = input.dims();
    DeviceBuffer<float> deviceInput(dims.voxelCount());
    DeviceBuffer<float> deviceOutput(dims.voxelCount());
    deviceInput.upload(input.voxels());

    const float ms = filter(deviceInput, deviceOutput, dims);
    if (gpuMs != nullptr) {
        *gpuMs = ms;
    }

    std::vector<float> voxels(dims.voxelCount());
    deviceOutput.download(voxels);
    Volume result(dims, std::move(voxels), input.spacing());
    result.setName(input.name());
    return result;
}

} // namespace

float gaussianBlur(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, Dims dims, float sigma,
                   ConvolutionKernel kernel) {
    validate(input, output, dims);
    const std::vector<float> weights = makeGaussianKernel(sigma);
    const int radius = static_cast<int>(weights.size() / 2);
    VS_CUDA_CHECK(cudaMemcpyToSymbol(c_weights, weights.data(), weights.size() * sizeof(float)));

    DeviceBuffer<float> temp(dims.voxelCount());
    GpuTimer timer;

    if (kernel == ConvolutionKernel::Naive) {
        const dim3 block(8, 8, 8);
        const dim3 grid = voxelGrid(dims, block);
        timer.start();
        convolveNaive<0><<<grid, block>>>(input.get(), output.get(), dims.x, dims.y, dims.z, radius);
        convolveNaive<1><<<grid, block>>>(output.get(), temp.get(), dims.x, dims.y, dims.z, radius);
        convolveNaive<2><<<grid, block>>>(temp.get(), output.get(), dims.x, dims.y, dims.z, radius);
    } else {
        const dim3 rowGrid(divUp(dims.x, kRowTile), static_cast<unsigned>(dims.y), static_cast<unsigned>(dims.z));
        const dim3 colBlock(kTileX, kBlockRows);
        const dim3 yGrid(divUp(dims.x, kTileX), divUp(dims.y, kColTile), static_cast<unsigned>(dims.z));
        const dim3 zGrid(divUp(dims.x, kTileX), divUp(dims.z, kColTile), static_cast<unsigned>(dims.y));
        timer.start();
        convolveTiledX<<<rowGrid, kRowTile>>>(input.get(), output.get(), dims.x, dims.y, radius);
        convolveTiledYZ<1><<<yGrid, colBlock>>>(output.get(), temp.get(), dims.x, dims.y, dims.z, radius);
        convolveTiledYZ<2><<<zGrid, colBlock>>>(temp.get(), output.get(), dims.x, dims.y, dims.z, radius);
    }
    VS_CUDA_CHECK(cudaGetLastError());
    return timer.stop();
}

float median3(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, Dims dims) {
    validate(input, output, dims);
    const dim3 block(8, 8, 4);
    GpuTimer timer;
    timer.start();
    median3Kernel<<<voxelGrid(dims, block), block>>>(input.get(), output.get(), dims.x, dims.y, dims.z);
    VS_CUDA_CHECK(cudaGetLastError());
    return timer.stop();
}

float sobelMagnitude(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, Dims dims) {
    validate(input, output, dims);
    const dim3 block(8, 8, 8);
    GpuTimer timer;
    timer.start();
    sobelKernel<<<voxelGrid(dims, block), block>>>(input.get(), output.get(), dims.x, dims.y, dims.z);
    VS_CUDA_CHECK(cudaGetLastError());
    return timer.stop();
}

Volume gaussianBlurGpu(const Volume& input, float sigma, float* gpuMs, ConvolutionKernel kernel) {
    return runOnGpu(input, gpuMs, [&](const DeviceBuffer<float>& in, DeviceBuffer<float>& out, Dims dims) {
        return gaussianBlur(in, out, dims, sigma, kernel);
    });
}

Volume median3Gpu(const Volume& input, float* gpuMs) {
    return runOnGpu(input, gpuMs, median3);
}

Volume sobelMagnitudeGpu(const Volume& input, float* gpuMs) {
    return runOnGpu(input, gpuMs, sobelMagnitude);
}

} // namespace vs::gpu
