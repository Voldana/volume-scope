// volumescope_bench: times every filter on the CPU (1 thread and all threads) and on the GPU,
// checks the GPU results against the CPU reference, and prints a Markdown table.
// With --stream FILE it benchmarks out-of-core rendering of that file instead.

#include "app/OrbitCamera.h"
#include "core/CpuFilters.h"
#include "core/Procedural.h"
#include "core/TransferFunction.h"
#include "core/VolumeSource.h"
#include "gpu/CudaCheck.h"
#include "gpu/DeviceBuffer.h"
#include "gpu/Filters.h"
#include "gpu/StreamingRenderer.h"
#include "gpu/TransferTexture.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    int size = 256;
    float sigma = 1.5f;
    int runs = 10;
    std::string streamPath; ///< Non-empty: benchmark out-of-core rendering of this file.
    std::size_t cacheMb = 256;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("Usage: volumescope_bench [--size N] [--sigma S] [--runs R]\n"
                                        "       volumescope_bench --stream volume_WxHxD_type.raw [--cache-mb MB]");
        }
        if (arg == "--size") {
            options.size = std::stoi(argv[++i]);
        } else if (arg == "--sigma") {
            options.sigma = std::stof(argv[++i]);
        } else if (arg == "--runs") {
            options.runs = std::stoi(argv[++i]);
        } else if (arg == "--stream") {
            options.streamPath = argv[++i];
        } else if (arg == "--cache-mb") {
            options.cacheMb = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else {
            throw std::invalid_argument("Unknown option " + arg);
        }
    }
    if (options.size < 8 || options.runs < 1) {
        throw std::invalid_argument("--size must be at least 8 and --runs at least 1");
    }
    return options;
}

template <typename Fn> double timeMs(Fn&& fn) {
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

/// Out-of-core rendering: index build, streaming until the first view is at full resolution,
/// then a full orbit that forces bricks to be evicted and streamed again.
int runStreaming(const Options& options) {
    constexpr int kWidth = 1280;
    constexpr int kHeight = 720;
    constexpr int kOrbitFrames = 120;
    constexpr double kMb = 1024.0 * 1024.0;

    cudaDeviceProp device{};
    VS_CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    const auto source = std::make_shared<vs::RawFileSource>(options.streamPath);
    const vs::TransferFunction transfer = vs::TransferFunction::preset(1);
    const vs::gpu::TransferTexture transferTexture(transfer);

    vs::gpu::StreamingConfig config;
    config.cacheBytes = options.cacheMb << 20;
    vs::gpu::StreamingRenderer streamer(source, transfer, config);
    streamer.resize(kWidth, kHeight);
    std::vector<std::uint32_t> pixels(static_cast<std::size_t>(kWidth) * kHeight);

    const vs::Dims dims = source->dims();
    const auto& s = streamer.stats();
    std::printf("GPU: %s | volume: %s %dx%dx%d, %.0f MB on disk | cache: %.0f MB (%d bricks of %d^3)\n", device.name,
                source->name().c_str(), dims.x, dims.y, dims.z, static_cast<double>(s.sourceBytes) / kMb,
                static_cast<double>(s.cacheBytes) / kMb, s.slotCount, streamer.layout().brickSize());
    std::printf("Bricks: %d total, %d empty under the Bone transfer function (never loaded)\n", s.bricksTotal,
                s.bricksEmpty);
    std::printf("Index build (brick value ranges + preview, one pass over the file): %.0f ms\n\n",
                static_cast<double>(s.indexBuildMs));

    vs::OrbitCamera camera;
    const auto paramsFor = [&](const vs::OrbitCamera& cam) {
        vs::gpu::RenderParams params;
        params.eye = cam.eye();
        cam.basis(params.forward, params.right, params.up);
        params.aspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);
        vs::gpu::fitVolume(params, dims, source->spacing());
        return params;
    };

    // 1. First view: preview only, then stream until every visible brick is resident.
    double uploadGpuMs = 0.0;
    double prepareMs = 0.0;
    int previousUploads = 0;
    const auto accumulate = [&] {
        if (previousUploads > 0) {
            uploadGpuMs += static_cast<double>(s.uploadMs); // uploadMs reports the previous batch.
        }
        prepareMs += static_cast<double>(s.prepareMs) * (s.bricksUploaded > 0 ? 1.0 : 0.0);
        previousUploads = s.bricksUploaded;
    };

    const auto start = std::chrono::steady_clock::now();
    const float previewMs = streamer.render(transferTexture.handle(), paramsFor(camera), pixels);
    accumulate();
    int frames = 1;
    while (!streamer.converged() && !s.saturated && frames < 1000) {
        (void)streamer.render(transferTexture.handle(), paramsFor(camera), pixels);
        accumulate();
        ++frames;
    }
    const double convergeMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const double firstViewMb = static_cast<double>(s.bytesUploaded) / kMb;

    std::vector<double> steady;
    for (int i = 0; i < 10; ++i) {
        steady.push_back(static_cast<double>(streamer.render(transferTexture.handle(), paramsFor(camera), pixels)));
    }
    std::sort(steady.begin(), steady.end());

    std::printf("| First view (%d x %d) | Value |\n|---|---|\n", kWidth, kHeight);
    std::printf("| Frame 1, drawn from the preview | %.2f ms |\n", static_cast<double>(previewMs));
    std::printf("| Frames until streaming settles | %d%s |\n", frames,
                s.saturated ? " (cache full, rest drawn from the preview)" : " (full resolution)");
    std::printf("| Wall time until streaming settles | %.0f ms |\n", convergeMs);
    std::printf("| Bricks streamed | %.0f MB (%.0f%% of the file as 16-bit) |\n", firstViewMb,
                100.0 * firstViewMb / (static_cast<double>(dims.voxelCount()) * 2.0 / kMb));
    std::printf("| CPU brick reads (all threads) | %.0f ms total |\n", prepareMs);
    std::printf("| GPU copy throughput (pinned, async) | %.1f GB/s |\n",
                uploadGpuMs > 0.0 ? firstViewMb / 1024.0 / (uploadGpuMs / 1000.0) : 0.0);
    std::printf("| Ray march at full resolution (median of 10) | %.2f ms |\n\n", steady[steady.size() / 2]);

    // 2. Orbit once around the volume: the cache must evict and re-stream as the view changes.
    const std::uint64_t before = s.bytesUploaded;
    std::vector<double> orbit;
    int saturatedFrames = 0;
    for (int i = 0; i < kOrbitFrames; ++i) {
        camera.rotate(2.0f * 3.14159265f / 0.005f / static_cast<float>(kOrbitFrames), 0.0f);
        orbit.push_back(static_cast<double>(streamer.render(transferTexture.handle(), paramsFor(camera), pixels)));
        saturatedFrames += s.saturated ? 1 : 0;
    }
    std::sort(orbit.begin(), orbit.end());
    std::printf("| Orbit, %d frames | Value |\n|---|---|\n", kOrbitFrames);
    std::printf("| Ray march, median / max | %.2f / %.2f ms |\n", orbit[orbit.size() / 2], orbit.back());
    std::printf("| Streamed during the orbit | %.0f MB |\n", static_cast<double>(s.bytesUploaded - before) / kMb);
    std::printf("| Frames where the cache was saturated | %d |\n", saturatedFrames);
    return 0;
}

int run(const Options& options) {
    if (!options.streamPath.empty()) {
        return runStreaming(options);
    }
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
    const double gaussSerial = timeMs([&] { reference = vs::gaussianBlurCpu(volume, options.sigma, CpuMode::Serial); });
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
    const double medianGpu =
        medianOfRuns(options.runs, [&] { return static_cast<double>(vs::gpu::median3(input, output, dims)); });
    printRow("Median 3x3x3", medianSerial, medianParallel, medianGpu, maxAbsDiff(reference, downloadResult()));

    // Sobel gradient magnitude.
    const double sobelSerial = timeMs([&] { reference = vs::sobelMagnitudeCpu(volume, CpuMode::Serial); });
    const double sobelParallel = timeMs([&] { (void)vs::sobelMagnitudeCpu(volume, CpuMode::Parallel); });
    const double sobelGpu =
        medianOfRuns(options.runs, [&] { return static_cast<double>(vs::gpu::sobelMagnitude(input, output, dims)); });
    printRow("Sobel gradient magnitude", sobelSerial, sobelParallel, sobelGpu, maxAbsDiff(reference, downloadResult()));

    // End-to-end cost including allocation and host <-> device copies, for honesty.
    const double roundTrip = medianOfRuns(
        options.runs, [&] { return timeMs([&] { (void)vs::gpu::gaussianBlurGpu(volume, options.sigma); }); });
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
