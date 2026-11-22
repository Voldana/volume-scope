#include "core/Procedural.h"

#include <algorithm>
#include <cmath>

namespace vs {
namespace {

float gaussianBand(float distance, float width) {
    const float t = distance / width;
    return std::exp(-t * t);
}

float toUnit(int i, int n) {
    return (static_cast<float>(i) + 0.5f) / static_cast<float>(n) * 2.0f - 1.0f;
}

} // namespace

Volume makeTestVolume(Dims dims) {
    Volume volume(dims);
    for (int z = 0; z < dims.z; ++z) {
        const float pz = toUnit(z, dims.z);
        for (int y = 0; y < dims.y; ++y) {
            const float py = toUnit(y, dims.y);
            for (int x = 0; x < dims.x; ++x) {
                const float px = toUnit(x, dims.x);
                const float r = std::sqrt(px * px + py * py + pz * pz);

                const float ripple = 0.75f + 0.25f * std::sin(10.0f * std::atan2(pz, px));
                const float shell = 0.4f * ripple * gaussianBand(r - 0.8f, 0.03f);

                const float ringRadius = std::sqrt(px * px + pz * pz) - 0.45f;
                const float ring = 0.7f * gaussianBand(std::sqrt(ringRadius * ringRadius + py * py), 0.1f);

                const float core = gaussianBand(std::max(r - 0.18f, 0.0f), 0.03f);

                volume.at(x, y, z) = std::clamp(std::max({shell, ring, core}), 0.0f, 1.0f);
            }
        }
    }
    volume.setName("procedural test volume");
    return volume;
}

} // namespace vs
