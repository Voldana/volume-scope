#include "gpu/VolumeTexture.h"

#include "gpu/CudaCheck.h"

#include <stdexcept>
#include <utility>

namespace vs::gpu {

VolumeTexture::VolumeTexture(const Volume& volume) : dims_(volume.dims()) {
    if (volume.empty()) {
        throw std::invalid_argument("Cannot create a texture from an empty volume");
    }
    create(volume.voxels().data(), cudaMemcpyHostToDevice);
}

VolumeTexture::VolumeTexture(const float* deviceVoxels, Dims dims) : dims_(dims) {
    if (deviceVoxels == nullptr || !dims.valid()) {
        throw std::invalid_argument("Invalid device voxels for VolumeTexture");
    }
    create(deviceVoxels, cudaMemcpyDeviceToDevice);
}

VolumeTexture::~VolumeTexture() {
    release();
}

VolumeTexture::VolumeTexture(VolumeTexture&& other) noexcept
    : array_(std::exchange(other.array_, nullptr)),
      texture_(std::exchange(other.texture_, 0)),
      dims_(std::exchange(other.dims_, {})) {}

VolumeTexture& VolumeTexture::operator=(VolumeTexture&& other) noexcept {
    if (this != &other) {
        release();
        array_ = std::exchange(other.array_, nullptr);
        texture_ = std::exchange(other.texture_, 0);
        dims_ = std::exchange(other.dims_, {});
    }
    return *this;
}

void VolumeTexture::create(const void* source, cudaMemcpyKind kind) {
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<float>();
    const cudaExtent extent = make_cudaExtent(static_cast<std::size_t>(dims_.x), static_cast<std::size_t>(dims_.y),
                                              static_cast<std::size_t>(dims_.z));
    VS_CUDA_CHECK(cudaMalloc3DArray(&array_, &format, extent));

    try {
        cudaMemcpy3DParms copy{};
        copy.srcPtr = make_cudaPitchedPtr(const_cast<void*>(source), static_cast<std::size_t>(dims_.x) * sizeof(float),
                                          static_cast<std::size_t>(dims_.x), static_cast<std::size_t>(dims_.y));
        copy.dstArray = array_;
        copy.extent = extent;
        copy.kind = kind;
        VS_CUDA_CHECK(cudaMemcpy3D(&copy));

        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = array_;

        cudaTextureDesc sampling{};
        sampling.addressMode[0] = cudaAddressModeClamp;
        sampling.addressMode[1] = cudaAddressModeClamp;
        sampling.addressMode[2] = cudaAddressModeClamp;
        sampling.filterMode = cudaFilterModeLinear;
        sampling.readMode = cudaReadModeElementType;
        sampling.normalizedCoords = 1;
        VS_CUDA_CHECK(cudaCreateTextureObject(&texture_, &resource, &sampling, nullptr));
    } catch (...) {
        release();
        throw;
    }
}

void VolumeTexture::release() noexcept {
    if (texture_ != 0) {
        cudaDestroyTextureObject(texture_);
        texture_ = 0;
    }
    if (array_ != nullptr) {
        cudaFreeArray(array_);
        array_ = nullptr;
    }
}

} // namespace vs::gpu
