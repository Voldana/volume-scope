#pragma once

#include <array>
#include <span>
#include <vector>

namespace vs {

struct Rgba {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
};

struct ControlPoint {
    float position = 0.0f; ///< Density in [0, 1].
    Rgba color;            ///< Colour and opacity per unit length.
};

/// Maps normalized density to colour and opacity. Control points are linearly
/// interpolated and baked into a lookup table that the GPU samples.
class TransferFunction {
public:
    static constexpr int kTableSize = 256;

    explicit TransferFunction(std::vector<ControlPoint> points);

    [[nodiscard]] Rgba sample(float density) const noexcept;
    [[nodiscard]] const std::array<Rgba, kTableSize>& table() const noexcept { return table_; }

    [[nodiscard]] static std::span<const char* const> presetNames() noexcept;
    [[nodiscard]] static TransferFunction preset(int index);

private:
    std::vector<ControlPoint> points_;
    std::array<Rgba, kTableSize> table_{};
};

} // namespace vs
