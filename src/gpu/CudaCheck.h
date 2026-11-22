#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace vs::gpu {

class CudaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline void check(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        throw CudaError(std::string(file) + ":" + std::to_string(line) + ": " + expression +
                        " failed: " + cudaGetErrorString(error));
    }
}

/// RAII pair of CUDA events for timing GPU work on the default stream.
class GpuTimer {
public:
    GpuTimer() {
        check(cudaEventCreate(&start_), "cudaEventCreate", __FILE__, __LINE__);
        if (const cudaError_t error = cudaEventCreate(&stop_); error != cudaSuccess) {
            cudaEventDestroy(start_);
            check(error, "cudaEventCreate", __FILE__, __LINE__);
        }
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start() { check(cudaEventRecord(start_), "cudaEventRecord", __FILE__, __LINE__); }

    /// Waits for the GPU and returns the elapsed time since start() in milliseconds.
    float stop() {
        check(cudaEventRecord(stop_), "cudaEventRecord", __FILE__, __LINE__);
        check(cudaEventSynchronize(stop_), "cudaEventSynchronize", __FILE__, __LINE__);
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime", __FILE__, __LINE__);
        return ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

} // namespace vs::gpu

#define VS_CUDA_CHECK(expr) ::vs::gpu::check((expr), #expr, __FILE__, __LINE__)
