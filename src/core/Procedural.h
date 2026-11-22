#pragma once

#include "core/Volume.h"

namespace vs {

/// Synthetic test volume (dense core, ring, rippled outer shell) so the viewer and
/// tests work without downloading a dataset.
[[nodiscard]] Volume makeTestVolume(Dims dims = {192, 192, 192});

} // namespace vs
