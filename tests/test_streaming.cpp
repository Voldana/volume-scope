#include "core/Bricking.h"
#include "core/Procedural.h"
#include "core/TransferFunction.h"
#include "core/VolumeSource.h"
#include "gpu/BrickAtlas.h"
#include "gpu/CudaHandles.h"
#include "gpu/Raymarcher.h"
#include "gpu/StreamingRenderer.h"
#include "gpu/TransferTexture.h"
#include "gpu/VolumeTexture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

bool hasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

#define REQUIRE_CUDA()                                                                                                 \
    if (!hasCudaDevice())                                                                                              \
    GTEST_SKIP() << "No CUDA device available"

constexpr int kImageSize = 96;

vs::gpu::RenderParams paramsFor(vs::Dims dims) {
    vs::gpu::RenderParams params; // Camera on +Z looking at the origin.
    vs::gpu::fitVolume(params, dims, {});
    return params;
}

/// Largest difference of any colour channel between two RGBA8 images.
int maxChannelDiff(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
    int result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (int shift = 0; shift < 24; shift += 8) {
            const int ca = static_cast<int>((a[i] >> shift) & 0xFFu);
            const int cb = static_cast<int>((b[i] >> shift) & 0xFFu);
            result = std::max(result, std::abs(ca - cb));
        }
    }
    return result;
}

std::vector<std::uint32_t> renderInCore(const vs::Volume& volume, const vs::TransferFunction& transfer) {
    const vs::gpu::VolumeTexture texture(volume);
    const vs::gpu::TransferTexture transferTexture(transfer);
    vs::gpu::Raymarcher raymarcher;
    raymarcher.resize(kImageSize, kImageSize);
    std::vector<std::uint32_t> pixels(kImageSize * kImageSize);
    (void)raymarcher.render(texture.handle(), transferTexture.handle(), paramsFor(volume.dims()), pixels);
    return pixels;
}

} // namespace

TEST(BrickAtlas, RoundTripsABrickThroughASlot) {
    REQUIRE_CUDA();
    // 10 slots: the grid holds at least 10 cells without wasting a whole layer.
    vs::gpu::BrickAtlas atlas(6, 10, 2);
    EXPECT_GE(atlas.slotGrid().voxelCount(), 10u);
    EXPECT_LE(atlas.slotGrid().voxelCount(), 12u);

    // Two bricks in one batch, into slots 7 and 2.
    constexpr std::size_t kVoxels = 6 * 6 * 6;
    vs::gpu::PinnedBuffer<std::uint16_t> bricks(2 * kVoxels);
    for (std::size_t i = 0; i < bricks.size(); ++i) {
        bricks.get()[i] = static_cast<std::uint16_t>(i * 101);
    }
    vs::gpu::PinnedBuffer<int> slots(2);
    slots.get()[0] = 7;
    slots.get()[1] = 2;
    const vs::gpu::CudaStream stream;
    atlas.uploadBatchAsync(bricks.span(), slots.span(), stream.get());
    stream.synchronize();

    std::vector<std::uint16_t> back(kVoxels);
    atlas.download(7, back);
    EXPECT_TRUE(std::equal(back.begin(), back.end(), bricks.get()));
    atlas.download(2, back);
    EXPECT_TRUE(std::equal(back.begin(), back.end(), bricks.get() + kVoxels));
}

TEST(StreamingRenderer, ConvergesToTheInCoreRender) {
    REQUIRE_CUDA();
    // Partial bricks on every axis.
    const vs::Volume volume = vs::makeTestVolume({70, 52, 45});
    const vs::TransferFunction transfer = vs::TransferFunction::preset(1);
    const vs::gpu::TransferTexture transferTexture(transfer);

    vs::gpu::StreamingConfig config;
    config.brickSize = 16;
    config.previewMaxDim = 16;
    vs::gpu::StreamingRenderer streamer(std::make_shared<vs::MemoryVolumeSource>(volume), transfer, config);
    streamer.resize(kImageSize, kImageSize);
    std::vector<std::uint32_t> pixels(kImageSize * kImageSize);

    (void)streamer.render(transferTexture.handle(), paramsFor(volume.dims()), pixels);
    EXPECT_GT(streamer.stats().bricksMissing, 0); // First frame comes from the preview.
    for (int frame = 0; frame < 20 && !streamer.converged(); ++frame) {
        (void)streamer.render(transferTexture.handle(), paramsFor(volume.dims()), pixels);
    }
    ASSERT_TRUE(streamer.converged());
    EXPECT_GT(streamer.stats().bricksEmpty, 0); // The corners of the test volume are empty.
    EXPECT_LT(streamer.stats().bricksResident, streamer.stats().bricksTotal - streamer.stats().bricksEmpty + 1);

    // Same samples, 16-bit instead of float storage: at most a couple of 8-bit levels apart.
    EXPECT_LE(maxChannelDiff(pixels, renderInCore(volume, transfer)), 2);
}

TEST(StreamingRenderer, SmallCacheSaturatesWithoutThrashing) {
    REQUIRE_CUDA();
    const vs::Volume volume = vs::makeTestVolume({64, 64, 64});
    const vs::TransferFunction transfer = vs::TransferFunction::preset(1);
    const vs::gpu::TransferTexture transferTexture(transfer);

    vs::gpu::StreamingConfig config;
    config.brickSize = 16;
    config.cacheBytes = 4 * 18 * 18 * 18 * sizeof(std::uint16_t); // Room for 4 bricks only.
    vs::gpu::StreamingRenderer streamer(std::make_shared<vs::MemoryVolumeSource>(volume), transfer, config);
    ASSERT_EQ(streamer.stats().slotCount, 4);
    streamer.resize(kImageSize, kImageSize);
    std::vector<std::uint32_t> pixels(kImageSize * kImageSize);

    for (int frame = 0; frame < 5; ++frame) {
        (void)streamer.render(transferTexture.handle(), paramsFor(volume.dims()), pixels);
        EXPECT_LE(streamer.stats().bricksResident, 4);
    }
    EXPECT_TRUE(streamer.stats().saturated);
    EXPECT_FALSE(streamer.converged());
    EXPECT_EQ(streamer.stats().bricksUploaded, 0); // Saturated: nothing left to evict, so no churn.
}

TEST(StreamingRenderer, SkipsBricksThatTheTransferFunctionHides) {
    REQUIRE_CUDA();
    const vs::Volume volume = vs::makeTestVolume({48, 48, 48});
    const vs::TransferFunction invisible({{0.0f, {1, 1, 1, 0}}, {1.0f, {1, 1, 1, 0}}});
    const vs::TransferFunction visible = vs::TransferFunction::preset(0);

    vs::gpu::StreamingConfig config;
    config.brickSize = 16;
    vs::gpu::StreamingRenderer streamer(std::make_shared<vs::MemoryVolumeSource>(volume), invisible, config);
    EXPECT_EQ(streamer.stats().bricksEmpty, streamer.stats().bricksTotal);

    streamer.resize(kImageSize, kImageSize);
    std::vector<std::uint32_t> pixels(kImageSize * kImageSize);
    const vs::gpu::TransferTexture invisibleTexture(invisible);
    (void)streamer.render(invisibleTexture.handle(), paramsFor(volume.dims()), pixels);
    EXPECT_TRUE(streamer.converged());
    EXPECT_EQ(streamer.stats().bytesUploaded, 0u);

    // Making the data visible must bring the bricks in.
    streamer.setTransferFunction(visible);
    EXPECT_LT(streamer.stats().bricksEmpty, streamer.stats().bricksTotal);
    const vs::gpu::TransferTexture visibleTexture(visible);
    (void)streamer.render(visibleTexture.handle(), paramsFor(volume.dims()), pixels);
    EXPECT_GT(streamer.stats().bricksUploaded, 0);
}
