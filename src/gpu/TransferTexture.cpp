#include "gpu/TransferTexture.h"

#include "gpu/CudaCheck.h"

#include <utility>

namespace vs::gpu {

static_assert(sizeof(Rgba) == sizeof(float4), "Rgba must match float4 layout");

TransferTexture::TransferTexture(const TransferFunction& transfer) {
    constexpr auto width = static_cast<std::size_t>(TransferFunction::kTableSize);
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
    VS_CUDA_CHECK(cudaMallocArray(&array_, &format, width));

    try {
        const std::size_t rowBytes = width * sizeof(float4);
        VS_CUDA_CHECK(cudaMemcpy2DToArray(array_, 0, 0, transfer.table().data(), rowBytes, rowBytes, 1,
                                          cudaMemcpyHostToDevice));

        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = array_;

        cudaTextureDesc sampling{};
        sampling.addressMode[0] = cudaAddressModeClamp;
        sampling.filterMode = cudaFilterModeLinear;
        sampling.readMode = cudaReadModeElementType;
        sampling.normalizedCoords = 1;
        VS_CUDA_CHECK(cudaCreateTextureObject(&texture_, &resource, &sampling, nullptr));
    } catch (...) {
        release();
        throw;
    }
}

TransferTexture::~TransferTexture() {
    release();
}

TransferTexture::TransferTexture(TransferTexture&& other) noexcept
    : array_(std::exchange(other.array_, nullptr)), texture_(std::exchange(other.texture_, 0)) {}

TransferTexture& TransferTexture::operator=(TransferTexture&& other) noexcept {
    if (this != &other) {
        release();
        array_ = std::exchange(other.array_, nullptr);
        texture_ = std::exchange(other.texture_, 0);
    }
    return *this;
}

void TransferTexture::release() noexcept {
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
