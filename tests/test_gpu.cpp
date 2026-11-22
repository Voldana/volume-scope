#include "core/CpuFilters.h"
#include "core/Procedural.h"
#include "core/TransferFunction.h"
#include "gpu/DeviceBuffer.h"
#include "gpu/Filters.h"
#include "gpu/Raymarcher.h"
#include "gpu/TransferTexture.h"
#include "gpu/VolumeTexture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

bool hasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

#define REQUIRE_CUDA()                                                                                                 \
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available"

} // namespace

TEST(DeviceBuffer, RoundTripsData) {
    REQUIRE_CUDA();
    const std::vector<float> source = {1.0f, 2.0f, 3.0f, 4.0f};
    vs::gpu::DeviceBuffer<float> buffer(source.size());
    buffer.upload(source);
    std::vector<float> result(source.size());
    buffer.download(result);
    EXPECT_EQ(result, source);
}

TEST(DeviceBuffer, MoveTransfersOwnership) {
    REQUIRE_CUDA();
    vs::gpu::DeviceBuffer<float> a(16);
    float* pointer = a.get();
    vs::gpu::DeviceBuffer<float> b(std::move(a));
    EXPECT_EQ(b.get(), pointer);
    EXPECT_EQ(b.size(), 16u);
    EXPECT_EQ(a.get(), nullptr); // NOLINT(bugprone-use-after-move): testing the moved-from state.
    EXPECT_EQ(a.size(), 0u);
}

TEST(DeviceBuffer, RejectsSizeMismatch) {
    REQUIRE_CUDA();
    vs::gpu::DeviceBuffer<float> buffer(4);
    const std::vector<float> tooSmall(3);
    EXPECT_THROW(buffer.upload(tooSmall), std::length_error);
}

TEST(GpuGaussian, MatchesCpuReference) {
    REQUIRE_CUDA();
    const vs::Volume volume = vs::makeTestVolume({48, 40, 32});
    const vs::Volume cpu = vs::gaussianBlurCpu(volume, 1.5f);
    const vs::Volume gpu = vs::gpu::gaussianBlurGpu(volume, 1.5f);
    ASSERT_EQ(cpu.dims(), gpu.dims());

    float maxError = 0.0f;
    const auto a = cpu.voxels();
    const auto b = gpu.voxels();
    for (std::size_t i = 0; i < a.size(); ++i) {
        maxError = std::max(maxError, std::fabs(a[i] - b[i]));
    }
    EXPECT_LT(maxError, 1e-5f);
}

TEST(GpuGaussian, RejectsMismatchedBuffers) {
    REQUIRE_CUDA();
    vs::gpu::DeviceBuffer<float> input(10);
    vs::gpu::DeviceBuffer<float> output(5);
    EXPECT_THROW(vs::gpu::gaussianBlur(input, output, vs::Dims{10, 1, 1}, 1.0f), std::invalid_argument);
}

TEST(Raymarcher, RendersVolumeOverBackground) {
    REQUIRE_CUDA();
    const vs::Volume volume = vs::makeTestVolume({32, 32, 32});
    const vs::gpu::VolumeTexture volumeTexture(volume);
    const vs::gpu::TransferTexture transferTexture(vs::TransferFunction::preset(0));

    constexpr int kSize = 64;
    vs::gpu::Raymarcher raymarcher;
    raymarcher.resize(kSize, kSize);
    std::vector<std::uint32_t> pixels(kSize * kSize);
    vs::gpu::RenderParams params; // Camera on +Z looking at the origin.
    params.stepSize = 1.0f / 64.0f;
    (void)raymarcher.render(volumeTexture.handle(), transferTexture.handle(), params, pixels);

    const std::uint32_t corner = pixels.front();                          // Ray misses the box.
    const std::uint32_t centre = pixels[(kSize / 2) * kSize + kSize / 2]; // Ray hits the dense core.
    EXPECT_NE(corner, centre);
    EXPECT_GT(centre & 0xFFu, corner & 0xFFu); // Brighter red channel where the volume is.
}
