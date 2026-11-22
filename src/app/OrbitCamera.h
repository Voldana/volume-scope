#pragma once

#include "core/Math.h"

#include <algorithm>
#include <cmath>

namespace vs {

/// Camera orbiting the origin: drag to rotate, scroll to zoom.
class OrbitCamera {
public:
    void rotate(float dxPixels, float dyPixels) noexcept {
        yaw_ -= dxPixels * kRotateSpeed;
        pitch_ = std::clamp(pitch_ + dyPixels * kRotateSpeed, -1.5f, 1.5f);
    }

    void zoom(float wheelSteps) noexcept {
        distance_ = std::clamp(distance_ * std::pow(0.9f, wheelSteps), 0.3f, 10.0f);
    }

    [[nodiscard]] Vec3 eye() const noexcept {
        const float c = std::cos(pitch_);
        return {distance_ * c * std::sin(yaw_), distance_ * std::sin(pitch_), distance_ * c * std::cos(yaw_)};
    }

    void basis(Vec3& forward, Vec3& right, Vec3& up) const noexcept {
        forward = normalize(-eye());
        right = normalize(cross(forward, Vec3{0.0f, 1.0f, 0.0f}));
        up = cross(right, forward);
    }

private:
    static constexpr float kRotateSpeed = 0.005f;
    float yaw_ = 0.6f;
    float pitch_ = 0.35f;
    float distance_ = 1.5f;
};

} // namespace vs
