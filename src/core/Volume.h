#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vs {

struct Dims {
    int x = 0;
    int y = 0;
    int z = 0;

    [[nodiscard]] constexpr std::size_t voxelCount() const noexcept {
        return static_cast<std::size_t>(x) * static_cast<std::size_t>(y) * static_cast<std::size_t>(z);
    }
    [[nodiscard]] constexpr bool valid() const noexcept { return x > 0 && y > 0 && z > 0; }

    friend constexpr bool operator==(const Dims&, const Dims&) = default;
};

/// Physical size of one voxel along each axis (any unit, only the ratios matter for rendering).
struct Spacing {
    float x = 1.0f;
    float y = 1.0f;
    float z = 1.0f;
};

/// Scalar volume with values normalized to [0, 1], stored x-fastest, then y, then z.
class Volume {
public:
    Volume() = default;
    explicit Volume(Dims dims, Spacing spacing = {});
    Volume(Dims dims, std::vector<float> voxels, Spacing spacing = {});

    [[nodiscard]] Dims dims() const noexcept { return dims_; }
    [[nodiscard]] Spacing spacing() const noexcept { return spacing_; }
    [[nodiscard]] bool empty() const noexcept { return voxels_.empty(); }
    [[nodiscard]] std::span<float> voxels() noexcept { return voxels_; }
    [[nodiscard]] std::span<const float> voxels() const noexcept { return voxels_; }
    [[nodiscard]] std::size_t sizeBytes() const noexcept { return voxels_.size() * sizeof(float); }

    [[nodiscard]] std::size_t index(int x, int y, int z) const noexcept {
        return (static_cast<std::size_t>(z) * static_cast<std::size_t>(dims_.y) + static_cast<std::size_t>(y)) *
                   static_cast<std::size_t>(dims_.x) +
               static_cast<std::size_t>(x);
    }
    [[nodiscard]] float& at(int x, int y, int z) noexcept { return voxels_[index(x, y, z)]; }
    [[nodiscard]] float at(int x, int y, int z) const noexcept { return voxels_[index(x, y, z)]; }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    void setName(std::string name) { name_ = std::move(name); }

private:
    Dims dims_{};
    Spacing spacing_{};
    std::vector<float> voxels_;
    std::string name_;
};

} // namespace vs
