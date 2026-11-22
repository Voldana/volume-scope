#pragma once

#include "core/Volume.h"

#include <cuda_runtime.h>

namespace vs::gpu {

/// A volume stored in a CUDA 3D array, sampled through a texture object with
/// trilinear filtering, normalized coordinates and clamp-to-edge addressing.
class VolumeTexture {
public:
    VolumeTexture() = default;
    explicit VolumeTexture(const Volume& volume);
    /// Creates the texture from voxels that already live in device memory (e.g. a filter output).
    VolumeTexture(const float* deviceVoxels, Dims dims);
    ~VolumeTexture();

    VolumeTexture(const VolumeTexture&) = delete;
    VolumeTexture& operator=(const VolumeTexture&) = delete;
    VolumeTexture(VolumeTexture&& other) noexcept;
    VolumeTexture& operator=(VolumeTexture&& other) noexcept;

    [[nodiscard]] cudaTextureObject_t handle() const noexcept { return texture_; }
    [[nodiscard]] Dims dims() const noexcept { return dims_; }

private:
    void create(const void* source, cudaMemcpyKind kind);
    void release() noexcept;

    cudaArray_t array_ = nullptr;
    cudaTextureObject_t texture_ = 0;
    Dims dims_{};
};

} // namespace vs::gpu
