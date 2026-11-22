#include "core/CpuFilters.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace vs {
namespace {

enum class Axis { X, Y, Z };

void convolveAxis(const Volume& input, Volume& output, std::span<const float> weights, Axis axis) {
    const Dims d = input.dims();
    const int radius = static_cast<int>(weights.size() / 2);
    for (int z = 0; z < d.z; ++z) {
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
    }
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

Volume gaussianBlurCpu(const Volume& input, float sigma) {
    const auto weights = makeGaussianKernel(sigma);
    Volume a(input.dims(), input.spacing());
    Volume b(input.dims(), input.spacing());
    convolveAxis(input, a, weights, Axis::X);
    convolveAxis(a, b, weights, Axis::Y);
    convolveAxis(b, a, weights, Axis::Z);
    a.setName(input.name());
    return a;
}

} // namespace vs
