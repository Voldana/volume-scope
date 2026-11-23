#pragma once

#include "core/Volume.h"

#include <vector>

namespace vs {

inline constexpr int kMaxGaussianRadius = 16;

/// Serial is the ground truth for the CUDA kernels; Parallel splits the work across all
/// hardware threads and is the fair baseline for benchmarks. Both give identical results.
enum class CpuMode { Serial, Parallel };

/// Normalized 1D Gaussian weights, radius = ceil(3 * sigma) clamped to [1, kMaxGaussianRadius].
/// Throws std::invalid_argument if sigma is not positive.
[[nodiscard]] std::vector<float> makeGaussianKernel(float sigma);

/// Separable 3D Gaussian blur (X, then Y, then Z pass, clamp-to-edge).
[[nodiscard]] Volume gaussianBlurCpu(const Volume& input, float sigma, CpuMode mode = CpuMode::Serial);

/// 3x3x3 median filter (clamp-to-edge). Removes speckle / impulse noise while keeping edges.
[[nodiscard]] Volume median3Cpu(const Volume& input, CpuMode mode = CpuMode::Serial);

/// 3D Sobel gradient magnitude (clamp-to-edge), normalized so a unit step edge gives 0.5.
/// Output is clamped to [0, 1].
[[nodiscard]] Volume sobelMagnitudeCpu(const Volume& input, CpuMode mode = CpuMode::Serial);

} // namespace vs
