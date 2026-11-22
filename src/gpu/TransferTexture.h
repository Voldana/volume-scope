#pragma once

#include "core/TransferFunction.h"

#include <cuda_runtime.h>

namespace vs::gpu {

/// A transfer function lookup table as a 1D float4 texture with linear filtering.
class TransferTexture {
public:
    TransferTexture() = default;
    explicit TransferTexture(const TransferFunction& transfer);
    ~TransferTexture();

    TransferTexture(const TransferTexture&) = delete;
    TransferTexture& operator=(const TransferTexture&) = delete;
    TransferTexture(TransferTexture&& other) noexcept;
    TransferTexture& operator=(TransferTexture&& other) noexcept;

    [[nodiscard]] cudaTextureObject_t handle() const noexcept { return texture_; }

private:
    void release() noexcept;

    cudaArray_t array_ = nullptr;
    cudaTextureObject_t texture_ = 0;
};

} // namespace vs::gpu
