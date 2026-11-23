#include "core/CpuFilters.h"

#include "core/Parallel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace vs {
namespace {

/// Runs body(z) for every slice, serially or across threads.
template <typename SliceBody>
void forEachSlice(int depth, CpuMode mode, SliceBody&& body) {
    if (mode == CpuMode::Serial) {
        for (int z = 0; z < depth; ++z) {
            body(z);
        }
        return;
    }
    parallelFor(depth, [&body](int begin, int end) {
        for (int z = begin; z < end; ++z) {
            body(z);
        }
    });
}

enum class Axis { X, Y, Z };

void convolveAxis(const Volume& input, Volume& output, std::span<const float> weights, Axis axis, CpuMode mode) {
    const Dims d = input.dims();
    const int radius = static_cast<int>(weights.size() / 2);
    forEachSlice(d.z, mode, [&](int z) {
        for (int y = 0; y < d.y; ++y) {
            for (int x = 0; x < d.x; ++x) {
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int sx = x;
                    int sy = y;
                    int sz = z;
                    switch (axis) {
                    case Axis::X:
                        sx = std::clamp(x + k, 0, d.x - 1);
                        break;
                    case Axis::Y:
                        sy = std::clamp(y + k, 0, d.y - 1);
                        break;
                    case Axis::Z:
                        sz = std::clamp(z + k, 0, d.z - 1);
                        break;
                    }
                    sum += weights[static_cast<std::size_t>(k + radius)] * input.at(sx, sy, sz);
                }
                output.at(x, y, z) = sum;
            }
        }
    });
}

/// Clamp-to-edge voxel read.
float sample(const Volume& v, int x, int y, int z) noexcept {
    const Dims d = v.dims();
    return v.at(std::clamp(x, 0, d.x - 1), std::clamp(y, 0, d.y - 1), std::clamp(z, 0, d.z - 1));
}

} // namespace

std::vector<float> makeGaussianKernel(float sigma) {
    if (!(sigma > 0.0f)) {
        throw std::invalid_argument("Gaussian sigma must be positive");
    }
    const int radius = std::clamp(static_cast<int>(std::ceil(3.0f * sigma)), 1, kMaxGaussianRadius);
    std::vector<float> weights(static_cast<std::size_t>(2 * radius + 1));
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float w = std::exp(-static_cast<float>(i * i) / (2.0f * sigma * sigma));
        weights[static_cast<std::size_t>(i + radius)] = w;
        sum += w;
    }
    for (float& w : weights) {
        w /= sum;
    }
    return weights;
}

Volume gaussianBlurCpu(const Volume& input, float sigma, CpuMode mode) {
    const auto weights = makeGaussianKernel(sigma);
    Volume a(input.dims(), input.spacing());
    Volume b(input.dims(), input.spacing());
    convolveAxis(input, a, weights, Axis::X, mode);
    convolveAxis(a, b, weights, Axis::Y, mode);
    convolveAxis(b, a, weights, Axis::Z, mode);
    a.setName(input.name());
    return a;
}

Volume median3Cpu(const Volume& input, CpuMode mode) {
    const Dims d = input.dims();
    Volume output(d, input.spacing());
    forEachSlice(d.z, mode, [&](int z) {
        std::array<float, 27> window{};
        for (int y = 0; y < d.y; ++y) {
            for (int x = 0; x < d.x; ++x) {
                std::size_t n = 0;
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            window[n++] = sample(input, x + dx, y + dy, z + dz);
                        }
                    }
                }
                std::nth_element(window.begin(), window.begin() + 13, window.end());
                output.at(x, y, z) = window[13];
            }
        }
    });
    output.setName(input.name());
    return output;
}

Volume sobelMagnitudeCpu(const Volume& input, CpuMode mode) {
    constexpr std::array<float, 3> kSmooth = {1.0f, 2.0f, 1.0f};
    constexpr float kNorm = 1.0f / 32.0f;
    const Dims d = input.dims();
    Volume output(d, input.spacing());
    forEachSlice(d.z, mode, [&](int z) {
        for (int y = 0; y < d.y; ++y) {
            for (int x = 0; x < d.x; ++x) {
                float gx = 0.0f;
                float gy = 0.0f;
                float gz = 0.0f;
                for (int a = -1; a <= 1; ++a) {
                    for (int b = -1; b <= 1; ++b) {
                        const float w =
                            kSmooth[static_cast<std::size_t>(a + 1)] * kSmooth[static_cast<std::size_t>(b + 1)];
                        gx += w * (sample(input, x + 1, y + a, z + b) - sample(input, x - 1, y + a, z + b));
                        gy += w * (sample(input, x + a, y + 1, z + b) - sample(input, x + a, y - 1, z + b));
                        gz += w * (sample(input, x + a, y + b, z + 1) - sample(input, x + a, y + b, z - 1));
                    }
                }
                gx *= kNorm;
                gy *= kNorm;
                gz *= kNorm;
                output.at(x, y, z) = std::min(std::sqrt(gx * gx + gy * gy + gz * gz), 1.0f);
            }
        }
    });
    output.setName(input.name());
    return output;
}

} // namespace vs
