#pragma once

#include "core/Math.h"
#include "gpu/DeviceBuffer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace vs::gpu {

struct RenderParams {
    Vec3 eye{0.0f, 0.0f, 2.0f};
    Vec3 forward{0.0f, 0.0f, -1.0f};
    Vec3 right{1.0f, 0.0f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float tanHalfFov = 0.4142f; ///< tan(fov / 2), 45 degrees by default.
    float aspect = 1.0f;

    Vec3 boxMin{-0.5f, -0.5f, -0.5f}; ///< World-space bounds of the volume.
    Vec3 boxMax{0.5f, 0.5f, 0.5f};

    float stepSize = 1.0f / 512.0f; ///< Ray step in world units.
    float densityScale = 50.0f;     ///< Multiplies transfer-function opacity.
    float brightness = 1.0f;
    int maxSteps = 4096;
    Vec3 background{0.06f, 0.06f, 0.08f};
};

/// Front-to-back emission/absorption ray marcher. Renders into a device image and
/// copies RGBA8 pixels (row 0 = top) back to the host.
class Raymarcher {
public:
    void resize(int width, int height);

    /// `pixels` must hold width * height values. Returns the kernel time in milliseconds.
    float render(cudaTextureObject_t volume, cudaTextureObject_t transfer, const RenderParams& params,
                 std::span<std::uint32_t> pixels);

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }

private:
    DeviceBuffer<std::uint32_t> image_;
    int width_ = 0;
    int height_ = 0;
};

} // namespace vs::gpu
