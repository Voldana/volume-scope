#pragma once

#include "core/Volume.h"

#include <vector>

namespace vs {

inline constexpr int kMaxGaussianRadius = 16;

/// Normalized 1D Gaussian weights, radius = ceil(3 * sigma) clamped to [1, kMaxGaussianRadius].
/// Throws std::invalid_argument if sigma is not positive.
[[nodiscard]] std::vector<float> makeGaussianKernel(float sigma);

/// Reference separable 3D Gaussian blur (X, then Y, then Z pass, clamp-to-edge).
/// Single-threaded on purpose: it is the ground truth for the CUDA version.
[[nodiscard]] Volume gaussianBlurCpu(const Volume& input, float sigma);

} // namespace vs
