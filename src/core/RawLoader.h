#pragma once

#include "core/Volume.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace vs {

enum class VoxelType { UInt8, UInt16, Float32 };

[[nodiscard]] std::size_t voxelSizeBytes(VoxelType type) noexcept;

struct RawInfo {
    std::string name;
    Dims dims;
    VoxelType type = VoxelType::UInt8;
};

/// Parses the Open SciVis naming convention, e.g. "bonsai_256x256x256_uint8.raw".
/// Directories and the extension are ignored. Returns nullopt if the name does not match.
[[nodiscard]] std::optional<RawInfo> parseRawFilename(std::string_view filename);

/// Loads a headerless raw volume and normalizes it to [0, 1]:
/// uint8 is divided by 255, uint16 and float32 are min/max normalized.
/// Throws std::runtime_error if the file is missing or has the wrong size.
[[nodiscard]] Volume loadRaw(const std::filesystem::path& path, const RawInfo& info);

/// Same as above, with dimensions and type taken from the file name.
[[nodiscard]] Volume loadRaw(const std::filesystem::path& path);

} // namespace vs
