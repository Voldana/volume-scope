#include "core/CpuFilters.h"
#include "core/Procedural.h"

#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>

TEST(GaussianKernel, IsNormalizedAndSymmetric) {
    const auto weights = vs::makeGaussianKernel(1.5f);
    ASSERT_EQ(weights.size() % 2, 1u);
    EXPECT_NEAR(std::accumulate(weights.begin(), weights.end(), 0.0f), 1.0f, 1e-6f);
    for (std::size_t i = 0; i < weights.size() / 2; ++i) {
        EXPECT_FLOAT_EQ(weights[i], weights[weights.size() - 1 - i]);
    }
}

TEST(GaussianKernel, RadiusIsClamped) {
    EXPECT_EQ(vs::makeGaussianKernel(100.0f).size(), static_cast<std::size_t>(2 * vs::kMaxGaussianRadius + 1));
    EXPECT_EQ(vs::makeGaussianKernel(0.01f).size(), 3u);
}

TEST(GaussianKernel, RejectsNonPositiveSigma) {
    EXPECT_THROW((void)vs::makeGaussianKernel(0.0f), std::invalid_argument);
    EXPECT_THROW((void)vs::makeGaussianKernel(-1.0f), std::invalid_argument);
}

TEST(GaussianBlurCpu, LeavesConstantVolumeUnchanged) {
    vs::Volume volume(vs::Dims{8, 6, 4});
    for (float& v : volume.voxels()) {
        v = 0.25f;
    }
    const vs::Volume blurred = vs::gaussianBlurCpu(volume, 1.0f);
    for (const float v : blurred.voxels()) {
        EXPECT_NEAR(v, 0.25f, 1e-6f);
    }
}

TEST(GaussianBlurCpu, PreservesMassOfInteriorImpulse) {
    vs::Volume volume(vs::Dims{17, 17, 17});
    volume.at(8, 8, 8) = 1.0f;
    const vs::Volume blurred = vs::gaussianBlurCpu(volume, 1.0f);
    const auto voxels = blurred.voxels();
    EXPECT_NEAR(std::accumulate(voxels.begin(), voxels.end(), 0.0f), 1.0f, 1e-5f);
    EXPECT_LT(blurred.at(8, 8, 8), 1.0f);
    EXPECT_FLOAT_EQ(blurred.at(7, 8, 8), blurred.at(9, 8, 8));
    EXPECT_FLOAT_EQ(blurred.at(8, 7, 8), blurred.at(8, 8, 9));
}

TEST(ProceduralVolume, StaysInUnitRange) {
    const vs::Volume volume = vs::makeTestVolume({24, 24, 24});
    for (const float v : volume.voxels()) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
}
