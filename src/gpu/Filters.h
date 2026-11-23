#pragma once

#include "core/Volume.h"
#include "gpu/DeviceBuffer.h"

namespace vs::gpu {

/// Naive: one thread per voxel reading straight from global memory.
/// Tiled: each block stages its neighbourhood in shared memory first.
enum class ConvolutionKernel { Naive, Tiled };

// All device-memory filters below: `input` and `output` must each hold dims.voxelCount() floats and
// must not alias. They return the GPU time of the kernels in milliseconds (transfers excluded).

/// Separable 3D Gaussian blur (X, then Y, then Z pass, clamp-to-edge), matching vs::gaussianBlurCpu.
float gaussianBlur(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, Dims dims, float sigma,
                   ConvolutionKernel kernel = ConvolutionKernel::Tiled);

/// 3x3x3 median filter, matching vs::median3Cpu exactly.
float median3(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, Dims dims);

/// 3D Sobel gradient magnitude, matching vs::sobelMagnitudeCpu.
float sobelMagnitude(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, Dims dims);

// Convenience wrappers: upload, filter on the GPU, download. `gpuMs` receives the kernel time.
[[nodiscard]] Volume gaussianBlurGpu(const Volume& input, float sigma, float* gpuMs = nullptr,
                                     ConvolutionKernel kernel = ConvolutionKernel::Tiled);
[[nodiscard]] Volume median3Gpu(const Volume& input, float* gpuMs = nullptr);
[[nodiscard]] Volume sobelMagnitudeGpu(const Volume& input, float* gpuMs = nullptr);

} // namespace vs::gpu
