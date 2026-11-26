#include "app/OrbitCamera.h"
#include "core/CpuFilters.h"
#include "core/Procedural.h"
#include "core/RawLoader.h"
#include "core/TransferFunction.h"
#include "core/VolumeSource.h"
#include "gpu/Filters.h"
#include "gpu/Raymarcher.h"
#include "gpu/StreamingRenderer.h"
#include "gpu/TransferTexture.h"
#include "gpu/VolumeTexture.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

namespace {

struct Settings {
    float densityScale = 50.0f;
    float brightness = 1.2f;
    float samplesPerVoxel = 1.0f;
    float renderScale = 1.0f;
    float sigma = 1.5f;
    int preset = 1;
    int filter = 0; ///< 0 Gaussian, 1 median, 2 Sobel.
};

/// OpenGL texture that shows the CUDA render. Uses only GL 1.1 entry points,
/// so no extension loader is needed.
// TODO(stage 1): replace the host round trip with CUDA-OpenGL interop (cudaGraphicsGLRegisterImage).
class DisplayTexture {
public:
    DisplayTexture() { glGenTextures(1, &id_); }
    ~DisplayTexture() { glDeleteTextures(1, &id_); }
    DisplayTexture(const DisplayTexture&) = delete;
    DisplayTexture& operator=(const DisplayTexture&) = delete;

    void upload(int width, int height, const std::vector<std::uint32_t>& pixels) {
        glBindTexture(GL_TEXTURE_2D, id_);
        if (width != width_ || height != height_) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
            width_ = width;
            height_ = height;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        }
    }

    [[nodiscard]] GLuint id() const noexcept { return id_; }

private:
    GLuint id_ = 0;
    int width_ = 0;
    int height_ = 0;
};

vs::gpu::RenderParams makeRenderParams(const vs::OrbitCamera& camera, vs::Dims dims, vs::Spacing spacing, int width,
                                       int height, const Settings& settings) {
    vs::gpu::RenderParams params;
    params.eye = camera.eye();
    camera.basis(params.forward, params.right, params.up);
    params.aspect = static_cast<float>(width) / static_cast<float>(height);
    vs::gpu::fitVolume(params, dims, spacing, settings.samplesPerVoxel);
    params.densityScale = settings.densityScale;
    params.brightness = settings.brightness;
    return params;
}

struct Options {
    std::string volumePath;     ///< Empty: use the procedural test volume.
    std::string screenshotPath; ///< Non-empty: save one frame as PNG and exit.
    int screenshotFrame = 10;   ///< Frames to render first, so the UI layout settles.
    int preset = 1;
    float blurSigma = 0.0f; ///< > 0: apply the GPU Gaussian at startup.
    float density = 50.0f;
    bool stream = false; ///< Out-of-core: stream bricks into a GPU cache instead of loading the volume.
    std::size_t cacheMb = 256;
};

constexpr const char* kUsage =
    "Usage: volumescope [volume_WxHxD_type.raw] [--preset N] [--density D] [--blur SIGMA]\n"
    "                   [--stream] [--cache-mb MB] [--screenshot out.png]\n"
    "  --stream     out-of-core rendering: bricks stream into a GPU cache of --cache-mb (default 256)\n";

/// Upper bound on frames rendered before a --screenshot is taken while bricks stream in.
constexpr int kMaxScreenshotFrames = 600;

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(arg + " needs a value\n" + kUsage);
            }
            return argv[++i];
        };
        if (arg == "--screenshot") {
            options.screenshotPath = next();
        } else if (arg == "--preset") {
            options.preset = std::stoi(next());
        } else if (arg == "--density") {
            options.density = std::stof(next());
        } else if (arg == "--blur") {
            options.blurSigma = std::stof(next());
        } else if (arg == "--stream") {
            options.stream = true;
        } else if (arg == "--cache-mb") {
            options.cacheMb = static_cast<std::size_t>(std::stoul(next()));
        } else if (arg == "--help" || arg == "-h") {
            std::printf("%s", kUsage);
            std::exit(0);
        } else if (!arg.empty() && arg.front() == '-') {
            throw std::invalid_argument("Unknown option " + arg + "\n" + kUsage);
        } else {
            options.volumePath = arg;
        }
    }
    const int presetCount = static_cast<int>(vs::TransferFunction::presetNames().size());
    if (options.preset < 0 || options.preset >= presetCount) {
        throw std::invalid_argument("--preset must be between 0 and " + std::to_string(presetCount - 1));
    }
    if (options.stream && options.blurSigma > 0.0f) {
        throw std::invalid_argument("--blur needs the volume in memory and cannot be combined with --stream");
    }
    if (options.stream && options.cacheMb == 0) {
        throw std::invalid_argument("--cache-mb must be positive");
    }
    return options;
}

vs::Volume loadInitialVolume(const Options& options) {
    if (!options.volumePath.empty()) {
        std::printf("Loading %s\n", options.volumePath.c_str());
        return vs::loadRaw(options.volumePath);
    }
    std::printf("No file given, using the procedural test volume.\n%s", kUsage);
    return vs::makeTestVolume();
}

/// Opens the volume for streaming: memory-mapped file, or the procedural volume held in memory.
std::shared_ptr<const vs::VolumeSource> openStreamSource(const Options& options) {
    if (!options.volumePath.empty()) {
        std::printf("Streaming %s (memory-mapped)\n", options.volumePath.c_str());
        return std::make_shared<vs::RawFileSource>(options.volumePath);
    }
    std::printf("No file given, streaming the procedural test volume.\n%s", kUsage);
    return std::make_shared<vs::MemoryVolumeSource>(vs::makeTestVolume());
}

/// Reads the back buffer (UI included) and writes it as an opaque PNG.
void saveScreenshot(const std::string& path, int width, int height) {
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    for (std::size_t i = 3; i < rgba.size(); i += 4) {
        rgba[i] = 255;
    }
    const int stride = width * 4;
    // OpenGL rows start at the bottom; PNG rows start at the top.
    stbi_flip_vertically_on_write(1);
    if (stbi_write_png(path.c_str(), width, height, 4, rgba.data(), stride) == 0) {
        throw std::runtime_error("Failed to write screenshot to '" + path + "'");
    }
    std::printf("Saved screenshot to %s\n", path.c_str());
}

struct GlfwSession {
    GlfwSession() {
        if (glfwInit() == GLFW_FALSE) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
    }
    ~GlfwSession() { glfwTerminate(); }
    GlfwSession(const GlfwSession&) = delete;
    GlfwSession& operator=(const GlfwSession&) = delete;
};

class WindowHandle {
public:
    explicit WindowHandle(GLFWwindow* window) : window_(window) {
        if (window_ == nullptr) {
            throw std::runtime_error("Failed to create an OpenGL 3.3 window");
        }
    }
    ~WindowHandle() { glfwDestroyWindow(window_); }
    WindowHandle(const WindowHandle&) = delete;
    WindowHandle& operator=(const WindowHandle&) = delete;

    [[nodiscard]] GLFWwindow* get() const noexcept { return window_; }

private:
    GLFWwindow* window_;
};

struct ImGuiSession {
    explicit ImGuiSession(GLFWwindow* window) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330 core");
    }
    ~ImGuiSession() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
    ImGuiSession(const ImGuiSession&) = delete;
    ImGuiSession& operator=(const ImGuiSession&) = delete;
};

int run(int argc, char** argv) {
    const Options options = parseOptions(argc, argv);

    // Either an in-memory volume (filters available) or a streamed source (out-of-core).
    std::shared_ptr<const vs::VolumeSource> streamSource;
    vs::Volume original;
    if (options.stream) {
        streamSource = openStreamSource(options);
    } else {
        original = loadInitialVolume(options);
    }
    vs::Volume current = original;

    GlfwSession glfw;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // Declared before every GL-owning object so the context outlives them.
    const WindowHandle windowHandle(glfwCreateWindow(1600, 900, "VolumeScope", nullptr, nullptr));
    GLFWwindow* window = windowHandle.get();
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    ImGuiSession imgui(window);

    Settings settings;
    settings.preset = options.preset;
    settings.densityScale = options.density;
    std::string status = "Ready";
    if (options.blurSigma > 0.0f) {
        settings.sigma = options.blurSigma;
        float gpuMs = 0.0f;
        current = vs::gpu::gaussianBlurGpu(current, options.blurSigma, &gpuMs);
        status = "GPU Gaussian: " + std::to_string(gpuMs) + " ms (kernels only)";
    }

    vs::OrbitCamera camera;
    vs::gpu::Raymarcher raymarcher;
    vs::gpu::TransferTexture transferTexture(vs::TransferFunction::preset(settings.preset));
    vs::gpu::VolumeTexture volumeTexture;
    std::unique_ptr<vs::gpu::StreamingRenderer> streamer;
    if (streamSource) {
        vs::gpu::StreamingConfig config;
        config.cacheBytes = options.cacheMb << 20;
        streamer = std::make_unique<vs::gpu::StreamingRenderer>(streamSource,
                                                                vs::TransferFunction::preset(settings.preset), config);
        const auto& s = streamer->stats();
        std::printf("%d bricks (%d empty under this transfer function), %d cache slots, index built in %.0f ms\n",
                    s.bricksTotal, s.bricksEmpty, s.slotCount, static_cast<double>(s.indexBuildMs));
    } else {
        volumeTexture = vs::gpu::VolumeTexture(current);
    }
    const vs::Dims dims = streamSource ? streamSource->dims() : current.dims();
    const vs::Spacing spacing = streamSource ? streamSource->spacing() : current.spacing();

    DisplayTexture display;
    std::vector<std::uint32_t> pixels;
    float renderMs = 0.0f;
    int renderWidth = 0;
    int renderHeight = 0;
    int frame = 0;

    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuiIO& io = ImGui::GetIO();

        if (!io.WantCaptureMouse) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                camera.rotate(io.MouseDelta.x, io.MouseDelta.y);
            }
            if (io.MouseWheel != 0.0f) {
                camera.zoom(io.MouseWheel);
            }
        }

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        if (framebufferWidth > 0 && framebufferHeight > 0) {
            renderWidth = std::max(1, static_cast<int>(static_cast<float>(framebufferWidth) * settings.renderScale));
            renderHeight = std::max(1, static_cast<int>(static_cast<float>(framebufferHeight) * settings.renderScale));
            pixels.resize(static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight));
            const vs::gpu::RenderParams params =
                makeRenderParams(camera, dims, spacing, renderWidth, renderHeight, settings);
            if (streamer) {
                streamer->resize(renderWidth, renderHeight);
                renderMs = streamer->render(transferTexture.handle(), params, pixels);
            } else {
                raymarcher.resize(renderWidth, renderHeight);
                renderMs = raymarcher.render(volumeTexture.handle(), transferTexture.handle(), params, pixels);
            }
            display.upload(renderWidth, renderHeight, pixels);
            ImGui::GetBackgroundDrawList()->AddImage(static_cast<ImTextureID>(display.id()), ImVec2(0.0f, 0.0f),
                                                     io.DisplaySize);
        }

        ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(470.0f, 0.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("VolumeScope");

        constexpr double kMb = 1024.0 * 1024.0;
        if (streamSource) {
            ImGui::Text("%s", streamSource->name().c_str());
            ImGui::Text("%d x %d x %d  (%.0f MB source, streamed)", dims.x, dims.y, dims.z,
                        static_cast<double>(streamSource->storedBytes()) / kMb);
        } else {
            ImGui::Text("%s", current.name().c_str());
            ImGui::Text("%d x %d x %d  (%.1f MB as float)", dims.x, dims.y, dims.z,
                        static_cast<double>(current.sizeBytes()) / kMb);
        }
        ImGui::Text("Ray march: %.2f ms at %d x %d  |  %.0f FPS", static_cast<double>(renderMs), renderWidth,
                    renderHeight, static_cast<double>(io.Framerate));

        ImGui::SeparatorText("Rendering");
        ImGui::SliderFloat("Density", &settings.densityScale, 1.0f, 400.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Brightness", &settings.brightness, 0.2f, 4.0f);
        ImGui::SliderFloat("Samples / voxel", &settings.samplesPerVoxel, 0.25f, 4.0f);
        ImGui::SliderFloat("Render scale", &settings.renderScale, 0.25f, 1.0f);
        const auto presets = vs::TransferFunction::presetNames();
        if (ImGui::Combo("Transfer function", &settings.preset, presets.data(), static_cast<int>(presets.size()))) {
            const vs::TransferFunction transfer = vs::TransferFunction::preset(settings.preset);
            transferTexture = vs::gpu::TransferTexture(transfer);
            if (streamer) {
                streamer->setTransferFunction(transfer); // Changes which bricks are empty.
            }
        }

        if (streamer) {
            const vs::gpu::StreamingStats& s = streamer->stats();
            ImGui::SeparatorText("Out-of-core streaming");
            ImGui::Text("Cache: %d / %d bricks resident (%.0f MB on GPU)", s.bricksResident, s.slotCount,
                        static_cast<double>(s.cacheBytes) / kMb);
            ImGui::Text("Empty bricks skipped: %d of %d", s.bricksEmpty, s.bricksTotal);
            if (streamer->converged()) {
                ImGui::TextUnformatted("Full resolution: every visible brick is resident");
            } else {
                ImGui::Text("Visible bricks still streaming: %d%s", s.bricksMissing,
                            s.saturated ? "  (cache full, rest uses preview)" : "");
            }
            ImGui::Text("Last batch: %d bricks | read %.1f ms (CPU) | copy %.2f ms (GPU)", s.bricksUploaded,
                        static_cast<double>(s.prepareMs), static_cast<double>(s.uploadMs));
            ImGui::Text("Streamed so far: %.0f MB | index built in %.0f ms", static_cast<double>(s.bytesUploaded) / kMb,
                        static_cast<double>(s.indexBuildMs));
        } else {
            ImGui::SeparatorText("Image processing");
            constexpr const char* kFilterNames[] = {"Gaussian blur", "Median 3x3x3", "Sobel edges"};
            ImGui::Combo("Filter", &settings.filter, kFilterNames, 3);
            if (settings.filter == 0) {
                ImGui::SliderFloat("Sigma", &settings.sigma, 0.5f, 5.0f);
            }
            const std::string filterName = kFilterNames[settings.filter];
            if (ImGui::Button("Apply on GPU")) {
                float gpuMs = 0.0f;
                switch (settings.filter) {
                case 0:
                    current = vs::gpu::gaussianBlurGpu(current, settings.sigma, &gpuMs);
                    break;
                case 1:
                    current = vs::gpu::median3Gpu(current, &gpuMs);
                    break;
                default:
                    current = vs::gpu::sobelMagnitudeGpu(current, &gpuMs);
                    break;
                }
                volumeTexture = vs::gpu::VolumeTexture(current);
                status = "GPU " + filterName + ": " + std::to_string(gpuMs) + " ms (kernels only)";
            }
            ImGui::SameLine();
            if (ImGui::Button("Apply on CPU")) {
                const auto start = std::chrono::steady_clock::now();
                switch (settings.filter) {
                case 0:
                    current = vs::gaussianBlurCpu(current, settings.sigma, vs::CpuMode::Parallel);
                    break;
                case 1:
                    current = vs::median3Cpu(current, vs::CpuMode::Parallel);
                    break;
                default:
                    current = vs::sobelMagnitudeCpu(current, vs::CpuMode::Parallel);
                    break;
                }
                const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
                volumeTexture = vs::gpu::VolumeTexture(current);
                status = "CPU " + filterName + ": " + std::to_string(elapsed.count()) + " ms (all threads)";
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                current = original;
                volumeTexture = vs::gpu::VolumeTexture(current);
                status = "Volume reset";
            }
            ImGui::TextWrapped("%s", status.c_str());
        }

        ImGui::SeparatorText("Controls");
        ImGui::TextUnformatted("Left drag: orbit    Scroll: zoom");
        ImGui::End();

        ImGui::Render();
        glViewport(0, 0, framebufferWidth, framebufferHeight);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (!options.screenshotPath.empty() && ++frame >= options.screenshotFrame) {
            // When streaming, wait until the view is at full resolution (or the cache is full).
            const bool settled = !streamer || streamer->converged() || streamer->stats().saturated;
            if (settled || frame >= kMaxScreenshotFrames) {
                saveScreenshot(options.screenshotPath, framebufferWidth, framebufferHeight);
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }
        glfwSwapBuffers(window);
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "VolumeScope error: %s\n", error.what());
        return 1;
    }
}
