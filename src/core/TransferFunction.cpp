#include "core/TransferFunction.h"

#include <algorithm>
#include <stdexcept>

namespace vs {
namespace {

Rgba lerp(const Rgba& a, const Rgba& b, float t) noexcept {
    return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t};
}

constexpr std::array<const char*, 3> kPresetNames = {"Grayscale", "Bone", "Heat"};

} // namespace

TransferFunction::TransferFunction(std::vector<ControlPoint> points) : points_(std::move(points)) {
    if (points_.empty()) {
        throw std::invalid_argument("A transfer function needs at least one control point");
    }
    std::sort(points_.begin(), points_.end(),
              [](const ControlPoint& a, const ControlPoint& b) { return a.position < b.position; });
    for (int i = 0; i < kTableSize; ++i) {
        table_[static_cast<std::size_t>(i)] = sample(static_cast<float>(i) / (kTableSize - 1));
    }
}

Rgba TransferFunction::sample(float density) const noexcept {
    density = std::clamp(density, 0.0f, 1.0f);
    if (density <= points_.front().position) {
        return points_.front().color;
    }
    if (density >= points_.back().position) {
        return points_.back().color;
    }
    const auto upper = std::upper_bound(points_.begin(), points_.end(), density,
                                        [](float d, const ControlPoint& p) { return d < p.position; });
    const auto lower = upper - 1;
    const float span = upper->position - lower->position;
    const float t = span > 0.0f ? (density - lower->position) / span : 0.0f;
    return lerp(lower->color, upper->color, t);
}

std::span<const char* const> TransferFunction::presetNames() noexcept {
    return kPresetNames;
}

TransferFunction TransferFunction::preset(int index) {
    switch (index) {
    case 0:
        return TransferFunction({{0.0f, {0, 0, 0, 0}}, {0.15f, {0, 0, 0, 0}}, {1.0f, {1, 1, 1, 1}}});
    case 1:
        return TransferFunction({{0.0f, {0, 0, 0, 0}},
                                 {0.25f, {0, 0, 0, 0}},
                                 {0.35f, {0.9f, 0.45f, 0.3f, 0.05f}},
                                 {0.6f, {1.0f, 0.9f, 0.75f, 0.4f}},
                                 {1.0f, {1, 1, 1, 0.9f}}});
    case 2:
        return TransferFunction({{0.0f, {0, 0, 0, 0}},
                                 {0.1f, {0.1f, 0.0f, 0.3f, 0.0f}},
                                 {0.4f, {0.8f, 0.1f, 0.1f, 0.2f}},
                                 {0.7f, {1.0f, 0.7f, 0.0f, 0.5f}},
                                 {1.0f, {1.0f, 1.0f, 0.8f, 0.9f}}});
    default:
        throw std::out_of_range("Unknown transfer function preset");
    }
}

} // namespace vs
