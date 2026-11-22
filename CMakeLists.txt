cmake_minimum_required(VERSION 3.28)

# Build for the GPU in this machine unless the caller picks architectures (CI does).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
endif()

project(VolumeScope VERSION 0.1.0 LANGUAGES CXX CUDA)

option(VOLUMESCOPE_BUILD_APP "Build the desktop viewer" ON)
option(VOLUMESCOPE_BUILD_TESTS "Build the unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
include(FetchContent)

# Warnings for our own C++ code only (third-party code is built without them).
add_library(volumescope_warnings INTERFACE)
if(MSVC)
    target_compile_options(volumescope_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
else()
    target_compile_options(volumescope_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
endif()

# ---------------------------------------------------------------------------
# Core: CPU-only data structures, loaders and reference algorithms.
# ---------------------------------------------------------------------------
add_library(volumescope_core STATIC
    src/core/Volume.cpp
    src/core/RawLoader.cpp
    src/core/Procedural.cpp
    src/core/TransferFunction.cpp
    src/core/CpuFilters.cpp)
target_include_directories(volumescope_core PUBLIC src)
target_link_libraries(volumescope_core PRIVATE volumescope_warnings)

# ---------------------------------------------------------------------------
# GPU: CUDA resources, kernels and the ray marcher.
# ---------------------------------------------------------------------------
add_library(volumescope_gpu STATIC
    src/gpu/VolumeTexture.cpp
    src/gpu/TransferTexture.cpp
    src/gpu/Raymarcher.cu
    src/gpu/Filters.cu)
target_link_libraries(volumescope_gpu
    PUBLIC volumescope_core CUDA::cudart
    PRIVATE volumescope_warnings)

# ---------------------------------------------------------------------------
# Desktop viewer: GLFW + Dear ImGui (OpenGL 3.3).
# ---------------------------------------------------------------------------
if(VOLUMESCOPE_BUILD_APP)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE)
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.8
        GIT_SHALLOW TRUE)
    FetchContent_Declare(stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG 6b5799bae8a1fe3792691ef4b72ae26434c3e799)
    FetchContent_MakeAvailable(glfw imgui stb)

    add_library(stb INTERFACE)
    target_include_directories(stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

    find_package(OpenGL REQUIRED)

    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
    target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

    add_executable(volumescope src/app/main.cpp)
    target_link_libraries(volumescope PRIVATE volumescope_gpu imgui stb volumescope_warnings)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT volumescope)
endif()

# ---------------------------------------------------------------------------
# Tests: CPU tests run anywhere; GPU tests are labelled "gpu" and skip
# themselves when no CUDA device is present.
# ---------------------------------------------------------------------------
if(VOLUMESCOPE_BUILD_TESTS)
    enable_testing()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)

    add_executable(volumescope_tests
        tests/test_raw_loader.cpp
        tests/test_cpu_filters.cpp
        tests/test_transfer_function.cpp)
    target_link_libraries(volumescope_tests PRIVATE volumescope_core GTest::gtest_main volumescope_warnings)
    gtest_discover_tests(volumescope_tests DISCOVERY_MODE PRE_TEST)

    add_executable(volumescope_gpu_tests tests/test_gpu.cpp)
    target_link_libraries(volumescope_gpu_tests PRIVATE volumescope_gpu GTest::gtest_main volumescope_warnings)
    gtest_discover_tests(volumescope_gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu)
endif()
