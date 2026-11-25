#include "core/BrickCache.h"
#include "core/Bricking.h"
#include "core/MappedFile.h"
#include "core/Procedural.h"
#include "core/RawLoader.h"
#include "core/TransferFunction.h"
#include "core/VolumeSource.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

std::filesystem::path writeTempFile(const std::string& filename, const void* data, std::size_t bytes) {
    auto dir = std::filesystem::temp_directory_path() / "volumescope_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / filename;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return path;
}

float unorm(std::uint16_t value) {
    return static_cast<float>(value) / 65535.0f;
}

} // namespace

// ---------------------------------------------------------------------------
// Memory-mapped files and volume sources
// ---------------------------------------------------------------------------

TEST(MappedFile, MapsFileContents) {
    const std::vector<std::uint8_t> bytes = {1, 2, 3, 250};
    const auto path = writeTempFile("mapped.bin", bytes.data(), bytes.size());
    const vs::MappedFile file(path);
    ASSERT_EQ(file.size(), 4u);
    EXPECT_EQ(static_cast<std::uint8_t>(file.bytes()[3]), 250);
}

TEST(MappedFile, MoveTransfersTheMapping) {
    const std::vector<std::uint8_t> bytes = {7, 8};
    const auto path = writeTempFile("mapped_move.bin", bytes.data(), bytes.size());
    vs::MappedFile a(path);
    const vs::MappedFile b(std::move(a));
    EXPECT_EQ(b.size(), 2u);
    EXPECT_EQ(a.size(), 0u); // NOLINT(bugprone-use-after-move): testing the moved-from state.
}

TEST(MappedFile, ThrowsForMissingFile) {
    EXPECT_THROW(vs::MappedFile("does/not/exist.raw"), std::runtime_error);
}

TEST(RawFileSource, MatchesLoadRawForUint8AndUint16) {
    const vs::Dims dims{5, 4, 3};
    std::vector<std::uint8_t> u8(dims.voxelCount());
    std::vector<std::uint16_t> u16(dims.voxelCount());
    for (std::size_t i = 0; i < u8.size(); ++i) {
        u8[i] = static_cast<std::uint8_t>((i * 37) % 256);
        u16[i] = static_cast<std::uint16_t>(1000 + i * 311);
    }
    const auto p8 = writeTempFile("src_5x4x3_uint8.raw", u8.data(), u8.size());
    const auto p16 = writeTempFile("src_5x4x3_uint16.raw", u16.data(), u16.size() * 2);

    for (const auto& path : {p8, p16}) {
        const vs::Volume reference = vs::loadRaw(path);
        const vs::RawFileSource source(path);
        ASSERT_EQ(source.dims(), dims);
        std::vector<float> row(3);
        for (int z = 0; z < dims.z; ++z) {
            for (int y = 0; y < dims.y; ++y) {
                source.readRow(1, y, z, row);
                for (int x = 0; x < 3; ++x) {
                    EXPECT_NEAR(row[static_cast<std::size_t>(x)], reference.at(x + 1, y, z), 1e-6f);
                }
            }
        }
    }
}

TEST(RawFileSource, RejectsWrongFileSize) {
    const std::vector<std::uint8_t> bytes(10);
    const auto path = writeTempFile("short_4x4x4_uint8.raw", bytes.data(), bytes.size());
    EXPECT_THROW(vs::RawFileSource{path}, std::runtime_error);
}

TEST(MemoryVolumeSource, RejectsRowsOutsideTheVolume) {
    const vs::MemoryVolumeSource source(vs::Volume(vs::Dims{4, 4, 4}));
    std::vector<float> row(3);
    EXPECT_THROW(source.readRow(2, 0, 0, row), std::out_of_range);
    EXPECT_THROW(source.readRow(0, 4, 0, row), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Brick layout, extraction, value ranges and preview
// ---------------------------------------------------------------------------

TEST(BrickLayout, CoversVolumeWithPartialBricks) {
    const vs::BrickLayout layout(vs::Dims{70, 32, 1}, 32);
    EXPECT_EQ(layout.grid(), (vs::Dims{3, 1, 1}));
    EXPECT_EQ(layout.count(), 3);
    EXPECT_EQ(layout.storedSize(), 34);
    for (int i = 0; i < layout.count(); ++i) {
        const vs::Dims c = layout.coords(i);
        EXPECT_EQ(layout.index(c.x, c.y, c.z), i);
    }
}

TEST(ExtractBrick, CopiesInteriorAndClampedApron) {
    // Awkward size: partial bricks on every axis.
    const vs::Volume volume = vs::makeTestVolume({37, 20, 9});
    const vs::MemoryVolumeSource source(volume);
    const vs::BrickLayout layout(volume.dims(), 8);
    std::vector<std::uint16_t> stored(layout.storedVoxels());
    const int s = layout.storedSize();

    for (int brick = 0; brick < layout.count(); ++brick) {
        vs::extractBrick(source, layout, brick, stored);
        const vs::Dims b = layout.coords(brick);
        for (int lz = 0; lz < s; ++lz) {
            for (int ly = 0; ly < s; ++ly) {
                for (int lx = 0; lx < s; ++lx) {
                    const int x = std::clamp(b.x * 8 - 1 + lx, 0, volume.dims().x - 1);
                    const int y = std::clamp(b.y * 8 - 1 + ly, 0, volume.dims().y - 1);
                    const int z = std::clamp(b.z * 8 - 1 + lz, 0, volume.dims().z - 1);
                    const std::size_t i = (static_cast<std::size_t>(lz) * s + ly) * s + lx;
                    ASSERT_NEAR(unorm(stored[i]), volume.at(x, y, z), 1.0f / 65535.0f)
                        << "brick " << brick << " local " << lx << "," << ly << "," << lz;
                }
            }
        }
    }
}

TEST(BrickRanges, ContainEveryVoxelOfBrickAndApron) {
    vs::Volume volume(vs::Dims{16, 8, 8});
    volume.at(8, 3, 3) = 1.0f; // First voxel of brick 1: also in brick 0's apron.
    const vs::MemoryVolumeSource source(volume);
    const vs::BrickLayout layout(volume.dims(), 8);
    const auto ranges = vs::computeBrickRanges(source, layout);
    ASSERT_EQ(ranges.size(), 2u);
    EXPECT_FLOAT_EQ(ranges[0].max, 1.0f);
    EXPECT_FLOAT_EQ(ranges[1].max, 1.0f);
    EXPECT_FLOAT_EQ(ranges[0].min, 0.0f);

    volume.at(8, 3, 3) = 0.0f;
    volume.at(15, 3, 3) = 0.5f; // Only inside brick 1.
    const auto ranges2 = vs::computeBrickRanges(vs::MemoryVolumeSource(volume), layout);
    EXPECT_FLOAT_EQ(ranges2[0].max, 0.0f);
    EXPECT_FLOAT_EQ(ranges2[1].max, 0.5f);
}

TEST(LodVolume, AveragesBlocksAndRoundsSizeUp) {
    vs::Volume volume(vs::Dims{10, 4, 4});
    for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 10; ++x) {
                volume.at(x, y, z) = x < 4 ? 1.0f : 0.0f;
            }
        }
    }
    EXPECT_EQ(vs::lodFactor(volume.dims(), 3), 4);
    const vs::Volume lod = vs::makeLodVolume(vs::MemoryVolumeSource(volume), 3);
    EXPECT_EQ(lod.dims(), (vs::Dims{3, 1, 1}));
    EXPECT_FLOAT_EQ(lod.at(0, 0, 0), 1.0f); // x 0..3
    EXPECT_FLOAT_EQ(lod.at(1, 0, 0), 0.0f); // x 4..7
    EXPECT_FLOAT_EQ(lod.at(2, 0, 0), 0.0f); // x 8..9 (partial block)
}

TEST(TransferVisibility, DetectsInvisibleValueRanges) {
    const vs::TransferFunction transfer({{0.0f, {0, 0, 0, 0}}, {0.5f, {0, 0, 0, 0}}, {1.0f, {1, 1, 1, 1}}});
    EXPECT_FLOAT_EQ(transfer.maxOpacity(0.0f, 0.3f), 0.0f);
    EXPECT_GT(transfer.maxOpacity(0.0f, 0.6f), 0.0f);
    EXPECT_GT(transfer.maxOpacity(0.9f, 1.0f), 0.0f);
}

// ---------------------------------------------------------------------------
// LRU brick cache
// ---------------------------------------------------------------------------

TEST(BrickCache, FillsFreeSlotsThenEvictsLeastRecentlyUsed) {
    vs::BrickCache cache(2);
    const auto a = cache.insert(10, 1);
    const auto b = cache.insert(11, 1);
    ASSERT_TRUE(a && b);
    EXPECT_FALSE(a->evicted);
    EXPECT_NE(a->slot, b->slot);

    cache.touch(10, 2); // 11 is now the least recently used.
    const auto c = cache.insert(12, 3);
    ASSERT_TRUE(c);
    ASSERT_TRUE(c->evicted);
    EXPECT_EQ(*c->evicted, 11);
    EXPECT_EQ(c->slot, b->slot);
    EXPECT_FALSE(cache.slotOf(11));
    EXPECT_EQ(cache.slotOf(10), a->slot);
    EXPECT_EQ(cache.residentCount(), 2);
}

TEST(BrickCache, NeverEvictsBricksUsedInTheCurrentFrame) {
    vs::BrickCache cache(1);
    ASSERT_TRUE(cache.insert(1, 5));
    EXPECT_FALSE(cache.insert(2, 5)); // Saturated: brick 1 is needed this frame.
    EXPECT_TRUE(cache.insert(2, 6));  // Next frame brick 1 may go.
}

TEST(BrickCache, RejectsDuplicateInsertAndClearFreesAll) {
    vs::BrickCache cache(3);
    ASSERT_TRUE(cache.insert(4, 0));
    EXPECT_THROW((void)cache.insert(4, 0), std::logic_error);
    cache.clear();
    EXPECT_EQ(cache.residentCount(), 0);
    EXPECT_FALSE(cache.slotOf(4));
}
