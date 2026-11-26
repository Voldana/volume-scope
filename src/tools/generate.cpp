// volumescope_gen: writes a procedural uint8 volume of any size, one slice at a time, so volumes
// larger than RAM or GPU memory can be created for out-of-core testing without a download.
//
//   volumescope_gen --size 1024 --out data      -> data/procedural_1024x1024x1024_uint8.raw (1 GB)

#include "core/Parallel.h"
#include "core/Procedural.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    int size = 1024;
    std::filesystem::path out = "data";
};

Options parseOptions(int argc, char** argv) {
    constexpr const char* kUsage = "Usage: volumescope_gen [--size N] [--out DIR]";
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument(kUsage);
        }
        if (arg == "--size") {
            options.size = std::stoi(argv[++i]);
        } else if (arg == "--out") {
            options.out = argv[++i];
        } else {
            throw std::invalid_argument("Unknown option " + arg + "\n" + kUsage);
        }
    }
    if (options.size < 8 || options.size > 4096) {
        throw std::invalid_argument("--size must be between 8 and 4096");
    }
    return options;
}

float toUnit(int i, int n) {
    return (static_cast<float>(i) + 0.5f) / static_cast<float>(n) * 2.0f - 1.0f;
}

/// The test volume plus fine surface detail, so full-resolution bricks look visibly sharper
/// than the low-resolution preview the streaming renderer shows while they load.
float detailedValue(float x, float y, float z) {
    const float base = vs::testVolumeValue(x, y, z);
    const float detail = 0.8f + 0.2f * std::sin(60.0f * x) * std::sin(60.0f * y) * std::sin(60.0f * z);
    return base * detail;
}

int run(const Options& options) {
    const int n = options.size;
    std::filesystem::create_directories(options.out);
    const std::string name =
        "procedural_" + std::to_string(n) + "x" + std::to_string(n) + "x" + std::to_string(n) + "_uint8.raw";
    const std::filesystem::path path = options.out / name;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot write '" + path.string() + "'");
    }

    const double gigabytes = static_cast<double>(n) * n * n / (1024.0 * 1024.0 * 1024.0);
    std::printf("Writing %s (%.2f GB)\n", path.string().c_str(), gigabytes);
    const auto start = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> slice(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int z = 0; z < n; ++z) {
        const float pz = toUnit(z, n);
        vs::parallelFor(n, [&](int begin, int end) {
            for (int y = begin; y < end; ++y) {
                const float py = toUnit(y, n);
                for (int x = 0; x < n; ++x) {
                    const float v = detailedValue(toUnit(x, n), py, pz);
                    slice[static_cast<std::size_t>(y) * static_cast<std::size_t>(n) + static_cast<std::size_t>(x)] =
                        static_cast<std::uint8_t>(std::lround(std::fmin(std::fmax(v, 0.0f), 1.0f) * 255.0f));
                }
            }
        });
        file.write(reinterpret_cast<const char*>(slice.data()), static_cast<std::streamsize>(slice.size()));
        if (!file) {
            throw std::runtime_error("Write failed (disk full?)");
        }
        if ((z + 1) % 64 == 0 || z + 1 == n) {
            std::printf("  %d / %d slices\n", z + 1, n);
        }
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("Done in %.1f s\n", seconds);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "volumescope_gen: %s\n", error.what());
        return 1;
    }
}
