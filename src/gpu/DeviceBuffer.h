#pragma once

#include "gpu/CudaCheck.h"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace vs::gpu {

/// Owning, move-only handle to a typed block of device memory.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            VS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void reset() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_); // Errors are ignored: destructors must not throw.
        }
        data_ = nullptr;
        count_ = 0;
    }

    [[nodiscard]] T* get() noexcept { return data_; }
    [[nodiscard]] const T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t sizeBytes() const noexcept { return count_ * sizeof(T); }

    void upload(std::span<const T> source) {
        if (source.size() != count_) {
            throw std::length_error("DeviceBuffer::upload: size mismatch");
        }
        VS_CUDA_CHECK(cudaMemcpy(data_, source.data(), sizeBytes(), cudaMemcpyHostToDevice));
    }

    void download(std::span<T> destination) const {
        if (destination.size() != count_) {
            throw std::length_error("DeviceBuffer::download: size mismatch");
        }
        VS_CUDA_CHECK(cudaMemcpy(destination.data(), data_, sizeBytes(), cudaMemcpyDeviceToHost));
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace vs::gpu
