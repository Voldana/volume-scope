#include "core/RawLoader.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

std::filesystem::path tempDir() {
    auto dir = std::filesystem::temp_directory_path() / "volumescope_tests";
    std::filesystem::create_directories(dir);
    return dir;
}

template <typename T>
std::filesystem::path writeRaw(const std::string& filename, const std::vector<T>& values) {
    const auto path = tempDir() / filename;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    return path;
}

} // namespace

TEST(ParseRawFilename, ParsesOpenSciVisConvention) {
    const auto info = vs::parseRawFilename("bonsai_256x256x128_uint8.raw");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->name, "bonsai");
    EXPECT_EQ(info->dims, (vs::Dims{256, 256, 128}));
    EXPECT_EQ(info->type, vs::VoxelType::UInt8);
}

TEST(ParseRawFilename, KeepsUnderscoresInNameAndIgnoresDirectories) {
    const auto info = vs::parseRawFilename("C:/data/head_mri_scan_64x32x16_uint16.raw");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->name, "head_mri_scan");
    EXPECT_EQ(info->dims, (vs::Dims{64, 32, 16}));
    EXPECT_EQ(info->type, vs::VoxelType::UInt16);
}

TEST(ParseRawFilename, AcceptsFloatAlias) {
    const auto info = vs::parseRawFilename("field_8x8x8_float.raw");
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->type, vs::VoxelType::Float32);
}

TEST(ParseRawFilename, RejectsMalformedNames) {
    EXPECT_FALSE(vs::parseRawFilename("volume.raw"));
    EXPECT_FALSE(vs::parseRawFilename("volume_256x256_uint8.raw"));
    EXPECT_FALSE(vs::parseRawFilename("volume_256x256x0_uint8.raw"));
    EXPECT_FALSE(vs::parseRawFilename("volume_256x256x256_int8.raw"));
    EXPECT_FALSE(vs::parseRawFilename("volume_axbxc_uint8.raw"));
}

TEST(LoadRaw, NormalizesUInt8ByFullRange) {
    const auto path = writeRaw<std::uint8_t>("u8_2x1x1_uint8.raw", {0, 255});
    const vs::Volume volume = vs::loadRaw(path);
    ASSERT_EQ(volume.dims(), (vs::Dims{2, 1, 1}));
    EXPECT_FLOAT_EQ(volume.at(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(volume.at(1, 0, 0), 1.0f);
    EXPECT_EQ(volume.name(), "u8");
}

TEST(LoadRaw, NormalizesUInt16ByMinMax) {
    const auto path = writeRaw<std::uint16_t>("u16_3x1x1_uint16.raw", {1000, 2000, 3000});
    const vs::Volume volume = vs::loadRaw(path);
    EXPECT_FLOAT_EQ(volume.at(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(volume.at(1, 0, 0), 0.5f);
    EXPECT_FLOAT_EQ(volume.at(2, 0, 0), 1.0f);
}

TEST(LoadRaw, UsesXFastestLayout) {
    const auto path = writeRaw<std::uint8_t>("layout_2x2x2_uint8.raw", {0, 1, 2, 3, 4, 5, 6, 255});
    const vs::Volume volume = vs::loadRaw(path);
    EXPECT_FLOAT_EQ(volume.at(1, 0, 0), 1.0f / 255.0f);
    EXPECT_FLOAT_EQ(volume.at(0, 1, 0), 2.0f / 255.0f);
    EXPECT_FLOAT_EQ(volume.at(0, 0, 1), 4.0f / 255.0f);
    EXPECT_FLOAT_EQ(volume.at(1, 1, 1), 1.0f);
}

TEST(LoadRaw, ThrowsWhenFileSizeDoesNotMatch) {
    const auto path = writeRaw<std::uint8_t>("short_4x4x4_uint8.raw", {1, 2, 3});
    EXPECT_THROW((void)vs::loadRaw(path), std::runtime_error);
}

TEST(LoadRaw, ThrowsWhenFileIsMissing) {
    EXPECT_THROW((void)vs::loadRaw(tempDir() / "missing_4x4x4_uint8.raw"), std::runtime_error);
}
