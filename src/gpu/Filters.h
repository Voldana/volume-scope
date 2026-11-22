#pragma once

#include "core/Volume.h"
#include "gpu/DeviceBuffer.h"

namespace vs::gpu {

/// Separable 3D Gaussian blur on device memory (X, then Y, then Z pass, clamp-to-edge),
/// matching vs::gaussianBlurCpu. `input` and `output` must each hold dims.voxelCount()
/// floats and must not alias. Returns the GPU time in milliseconds.
float gaussianBlur(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, Dims dims, float sigma);

/// Convenience wrapper: upload, blur on the GPU, download. `gpuMs` receives the kernel time.
[[nodiscard]] Volume gaussianBlurGpu(const Volume& input, float sigma, float* gpuMs = nullptr);

} // namespace vs::gpu
