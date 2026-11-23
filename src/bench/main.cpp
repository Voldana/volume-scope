// volumescope_bench: times every filter on the CPU (1 thread and all threads) and on the GPU,
// checks the GPU results against the CPU reference, and prints a Markdown table.

#include "core/CpuFilters.h"
#include "core/Procedural.h"
#include "gpu/CudaCheck.h"
#include "gpu/DeviceBuffer.h"
#include "gpu/Filters.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    int size = 256;
    float sigma = 1.5f;
    int runs = 10;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("Usage: volumescope_bench [--size N] [--sigma S] [--runs R]");
        }
        if (arg == "--size") {
            options.size = std::stoi(argv[++i]);
        } else if (arg == "--sigma") {
            options.sigma = std::stof(argv[++i]);
        } else if (arg == "--runs") {
            options.runs = std::stoi(argv[++i]);
        } else {
            throw std::invalid_argument("Unknown option " + arg);
        }
    }
    if (options.size < 8 || options.runs < 1) {
        throw std::invalid_argument("--size must be at least 8 and --runs at least 1");
    }
    return options;
}

template <typename Fn>
double timeMs(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

/// One warm-up call, then the median of `runs` timed calls.
double medianOfRuns(int runs, const std::function<double()>& run) {
    (void)run();
    std::vector<double> times;
    for (int i = 0; i < runs; ++i) {
        times.push_back(run());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

float maxAbsDiff(const vs::Volume& a, const vs::Volume& b) {
    float result = 0.0f;
    const auto va = a.voxels();
    const auto vb = b.voxels();
    for (std::size_t i = 0; i < va.size(); ++i) {
        result = std::max(result, std::fabs(va[i] - vb[i]));
    }
    return result;
}

void printRow(const std::string& name, double serialMs, double parallelMs, double gpuMs, float error) {
    std::printf("| %s | %.1f | %.1f | %.3f | %.0fx | %.0fx | %.1e |\n", name.c_str(), serialMs, parallelMs, gpuMs,
                serialMs / gpuMs, parallelMs / gpuMs, static_cast<double>(error));
}

int run(const Options& options) {
    const vs::Dims dims{options.size, options.size, options.size};
    const vs::Volume volume = vs::makeTestVolume(dims);

    cudaDeviceProp device{};
    VS_CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    std::printf("GPU: %s | CPU threads: %u | volume: %d^3 float (%.0f MB)\n", device.name,
                std::thread::hardware_concurrency(), options.size,
                static_cast<double>(volume.sizeBytes()) / (1024.0 * 1024.0));
    std::printf("CPU: one run each. GPU: median of %d runs, kernel time only. Error: max |GPU - CPU|.\n\n",
                options.runs);
    std::printf("| Filter | CPU 1 thread (ms) | CPU %u threads (ms) | GPU (ms) | vs 1 thread | vs all threads | "
                "Max error |\n",
                std::thread::hardware_concurrency());
    std::printf("|---|---|---|---|---|---|---|\n");

    vs::gpu::DeviceBuffer<float> input(dims.voxelCount());
    vs::gpu::DeviceBuffer<float> output(dims.voxelCount());
    input.upload(volume.voxels());
    std::vector<float> host(dims.voxelCount());
    const auto downloadResult = [&] {
        output.download(host);
        return vs::Volume(dims, host);
    };

    using vs::CpuMode;
    using vs::gpu::ConvolutionKernel;
    char label[64];

    // Gaussian: naive and tiled kernels against the same CPU numbers.
    vs::Volume reference;
    const double gaussSerial =
        timeMs([&] { reference = vs::gaussianBlurCpu(volume, options.sigma, CpuMode::Serial); });
    const double gaussParallel = timeMs([&] { (void)vs::gaussianBlurCpu(volume, options.sigma, CpuMode::Parallel); });
    for (const auto kernel : {ConvolutionKernel::Naive, ConvolutionKernel::Tiled}) {
        const double gpu = medianOfRuns(options.runs, [&] {
            return static_cast<double>(vs::gpu::gaussianBlur(input, output, dims, options.sigma, kernel));
        });
        std::snprintf(label, sizeof(label), "Gaussian, sigma %.1f (%s kernel)", static_cast<double>(options.sigma),
                      kernel == ConvolutionKernel::Naive ? "naive" : "tiled");
        printRow(label, gaussSerial, gaussParallel, gpu, maxAbsDiff(reference, downloadResult()));
    }

    // Median 3x3x3.
    const double medianSerial = timeMs([&] { reference = vs::median3Cpu(volume, CpuMode::Serial); });
    const double medianParallel = timeMs([&] { (void)vs::median3Cpu(volume, CpuMode::Parallel); });
    const double medianGpu = medianOfRuns(
        options.runs, [&] { return static_cast<double>(vs::gpu::median3(input, output, dims)); });
    printRow("Median 3x3x3", medianSerial, medianParallel, medianGpu, maxAbsDiff(reference, downloadResult()));

    // Sobel gradient magnitude.
    const double sobelSerial = timeMs([&] { reference = vs::sobelMagnitudeCpu(volume, CpuMode::Serial); });
    const double sobelParallel = timeMs([&] { (void)vs::sobelMagnitudeCpu(volume, CpuMode::Parallel); });
    const double sobelGpu = medianOfRuns(
        options.runs, [&] { return static_cast<double>(vs::gpu::sobelMagnitude(input, output, dims)); });
    printRow("Sobel gradient magnitude", sobelSerial, sobelParallel, sobelGpu,
             maxAbsDiff(reference, downloadResult()));

    // End-to-end cost including allocation and host <-> device copies, for honesty.
    const double roundTrip =
        medianOfRuns(options.runs, [&] { return timeMs([&] { (void)vs::gpu::gaussianBlurGpu(volume, options.sigma); }); });
    std::printf("\nTiled Gaussian including allocation and host<->device copies: %.1f ms\n", roundTrip);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "volumescope_bench: %s\n", error.what());
        return 1;
    }
}
