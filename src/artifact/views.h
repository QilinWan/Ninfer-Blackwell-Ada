#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/weight_view.h"

namespace ninfer::artifact {

// Resolves a logical parameter on `device`. A part of a Rows/Columns shard maps to the elements
// that device holds, and the view's shape narrows the same axis: the leading logical axis for a
// Rows shard, the parent row width for a Columns shard. Complete parents keep the logical shape.
[[nodiscard]] WeightView bind_view(const ParameterReference& reference,
                                   const MaterializedArtifact& materialized, int device = 0);

} // namespace ninfer::artifact
