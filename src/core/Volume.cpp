#include "core/Volume.h"

#include <stdexcept>

namespace vs {

Volume::Volume(Dims dims, Spacing spacing) : dims_(dims), spacing_(spacing) {
    if (!dims.valid()) {
        throw std::invalid_argument("Volume dimensions must be positive");
    }
    voxels_.assign(dims.voxelCount(), 0.0f);
}

Volume::Volume(Dims dims, std::vector<float> voxels, Spacing spacing)
    : dims_(dims), spacing_(spacing), voxels_(std::move(voxels)) {
    if (!dims.valid()) {
        throw std::invalid_argument("Volume dimensions must be positive");
    }
    if (voxels_.size() != dims.voxelCount()) {
        throw std::invalid_argument("Voxel count does not match volume dimensions");
    }
}

} // namespace vs
