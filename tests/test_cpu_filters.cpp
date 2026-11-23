#include "core/CpuFilters.h"
#include "core/Procedural.h"

#include <gtest/gtest.h>

#include <algorithm>
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

TEST(CpuFilters, ParallelMatchesSerialExactly) {
    const vs::Volume volume = vs::makeTestVolume({31, 23, 19});
    const auto equal = [](const vs::Volume& a, const vs::Volume& b) {
        return std::equal(a.voxels().begin(), a.voxels().end(), b.voxels().begin());
    };
    EXPECT_TRUE(equal(vs::gaussianBlurCpu(volume, 1.5f, vs::CpuMode::Serial),
                      vs::gaussianBlurCpu(volume, 1.5f, vs::CpuMode::Parallel)));
    EXPECT_TRUE(equal(vs::median3Cpu(volume, vs::CpuMode::Serial), vs::median3Cpu(volume, vs::CpuMode::Parallel)));
    EXPECT_TRUE(equal(vs::sobelMagnitudeCpu(volume, vs::CpuMode::Serial),
                      vs::sobelMagnitudeCpu(volume, vs::CpuMode::Parallel)));
}

TEST(Median3Cpu, RemovesIsolatedImpulse) {
    vs::Volume volume(vs::Dims{7, 7, 7});
    volume.at(3, 3, 3) = 1.0f;
    const vs::Volume filtered = vs::median3Cpu(volume);
    for (const float v : filtered.voxels()) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(Median3Cpu, KeepsStepEdge) {
    vs::Volume volume(vs::Dims{8, 4, 4});
    for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 4; x < 8; ++x) {
                volume.at(x, y, z) = 1.0f;
            }
        }
    }
    const vs::Volume filtered = vs::median3Cpu(volume);
    EXPECT_FLOAT_EQ(filtered.at(3, 2, 2), 0.0f);
    EXPECT_FLOAT_EQ(filtered.at(4, 2, 2), 1.0f);
}

TEST(SobelCpu, IsZeroOnConstantVolume) {
    vs::Volume volume(vs::Dims{6, 6, 6});
    for (float& v : volume.voxels()) {
        v = 0.7f;
    }
    // Keep the result alive: ranging over a temporary's span would dangle (C++20 range-for lifetime rule).
    const vs::Volume gradient = vs::sobelMagnitudeCpu(volume);
    for (const float v : gradient.voxels()) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(SobelCpu, GivesHalfOnUnitStepEdge) {
    vs::Volume volume(vs::Dims{8, 5, 5});
    for (int z = 0; z < 5; ++z) {
        for (int y = 0; y < 5; ++y) {
            for (int x = 4; x < 8; ++x) {
                volume.at(x, y, z) = 1.0f;
            }
        }
    }
    const vs::Volume gradient = vs::sobelMagnitudeCpu(volume);
    EXPECT_FLOAT_EQ(gradient.at(3, 2, 2), 0.5f);
    EXPECT_FLOAT_EQ(gradient.at(4, 2, 2), 0.5f);
    EXPECT_FLOAT_EQ(gradient.at(1, 2, 2), 0.0f);
}

TEST(ProceduralVolume, StaysInUnitRange) {
    const vs::Volume volume = vs::makeTestVolume({24, 24, 24});
    for (const float v : volume.voxels()) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
}
