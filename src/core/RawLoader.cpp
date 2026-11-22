#include "core/RawLoader.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace vs {
namespace {

std::optional<int> parsePositiveInt(std::string_view text) {
    int value = 0;
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    if (ec != std::errc{} || ptr != end || value <= 0) {
        return std::nullopt;
    }
    return value;
}

std::optional<Dims> parseDims(std::string_view text) {
    const auto first = text.find('x');
    if (first == std::string_view::npos) {
        return std::nullopt;
    }
    const auto second = text.find('x', first + 1);
    if (second == std::string_view::npos) {
        return std::nullopt;
    }
    const auto x = parsePositiveInt(text.substr(0, first));
    const auto y = parsePositiveInt(text.substr(first + 1, second - first - 1));
    const auto z = parsePositiveInt(text.substr(second + 1));
    if (!x || !y || !z) {
        return std::nullopt;
    }
    return Dims{*x, *y, *z};
}

std::optional<VoxelType> parseType(std::string_view text) {
    if (text == "uint8") {
        return VoxelType::UInt8;
    }
    if (text == "uint16") {
        return VoxelType::UInt16;
    }
    if (text == "float32" || text == "float") {
        return VoxelType::Float32;
    }
    return std::nullopt;
}

template <typename T>
std::vector<float> convertToFloat(const std::vector<std::byte>& bytes, std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        T value{};
        std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
        values[i] = static_cast<float>(value);
    }
    return values;
}

void normalizeMinMax(std::vector<float>& values) {
    if (values.empty()) {
        return;
    }
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    const float minValue = *lo;
    const float range = *hi - *lo;
    if (range <= 0.0f) {
        std::fill(values.begin(), values.end(), 0.0f);
        return;
    }
    for (float& value : values) {
        value = (value - minValue) / range;
    }
}

} // namespace

std::size_t voxelSizeBytes(VoxelType type) noexcept {
    switch (type) {
    case VoxelType::UInt8:
        return 1;
    case VoxelType::UInt16:
        return 2;
    case VoxelType::Float32:
        return 4;
    }
    return 0;
}

std::optional<RawInfo> parseRawFilename(std::string_view filename) {
    if (const auto slash = filename.find_last_of("/\\"); slash != std::string_view::npos) {
        filename.remove_prefix(slash + 1);
    }
    if (const auto dot = filename.rfind('.'); dot != std::string_view::npos) {
        filename = filename.substr(0, dot);
    }

    const auto typeSep = filename.rfind('_');
    if (typeSep == std::string_view::npos || typeSep == 0) {
        return std::nullopt;
    }
    const auto dimsSep = filename.rfind('_', typeSep - 1);
    if (dimsSep == std::string_view::npos) {
        return std::nullopt;
    }

    const auto type = parseType(filename.substr(typeSep + 1));
    const auto dims = parseDims(filename.substr(dimsSep + 1, typeSep - dimsSep - 1));
    if (!type || !dims) {
        return std::nullopt;
    }
    return RawInfo{std::string(filename.substr(0, dimsSep)), *dims, *type};
}

Volume loadRaw(const std::filesystem::path& path, const RawInfo& info) {
    if (!info.dims.valid()) {
        throw std::runtime_error("Invalid volume dimensions for '" + path.string() + "'");
    }
    const std::size_t count = info.dims.voxelCount();
    const std::size_t expectedBytes = count * voxelSizeBytes(info.type);

    std::error_code ec;
    const auto actualBytes = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("Cannot read '" + path.string() + "': " + ec.message());
    }
    if (actualBytes != expectedBytes) {
        throw std::runtime_error("'" + path.string() + "' has " + std::to_string(actualBytes) +
                                 " bytes, expected " + std::to_string(expectedBytes));
    }

    // TODO(stage 3): memory-map large files and stream bricks instead of reading everything.
    std::ifstream file(path, std::ios::binary);
    std::vector<std::byte> bytes(expectedBytes);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(expectedBytes))) {
        throw std::runtime_error("Failed to read '" + path.string() + "'");
    }

    std::vector<float> voxels;
    switch (info.type) {
    case VoxelType::UInt8:
        voxels = convertToFloat<std::uint8_t>(bytes, count);
        for (float& value : voxels) {
            value /= 255.0f;
        }
        break;
    case VoxelType::UInt16:
        voxels = convertToFloat<std::uint16_t>(bytes, count);
        normalizeMinMax(voxels);
        break;
    case VoxelType::Float32:
        voxels = convertToFloat<float>(bytes, count);
        normalizeMinMax(voxels);
        break;
    }

    Volume volume(info.dims, std::move(voxels));
    volume.setName(info.name);
    return volume;
}

Volume loadRaw(const std::filesystem::path& path) {
    const auto info = parseRawFilename(path.filename().string());
    if (!info) {
        throw std::runtime_error("Cannot infer size and type from '" + path.filename().string() +
                                 "'. Expected a name like bonsai_256x256x256_uint8.raw");
    }
    return loadRaw(path, *info);
}

} // namespace vs
