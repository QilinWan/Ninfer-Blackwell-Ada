#pragma once

#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {

// The fused query|key|gate|value projection writes four independent sections in parent-row order
// query|key|gate|value. The [14336,5120] parent covers 24 query and 4 key/value heads per section
// pair (6144/1024); under two-device tensor parallelism each rank's standalone [7168,5120] shard
// stacks its own 12 query/gate and 2 key/value heads (3072/512) in the same order, so each rank
// writes its own heads with no communication. The shard keeps the parent's K, so its routes, their
// token cutoffs and the workspace capacity are the parent's: only these section rows differ.
using Nvfp4AttentionInputOutput      = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;
using Nvfp4AttentionInputShardOutput = LinearBf16SegmentedOutput<3072, 512, 3072, 512>;

// Both problems are registered; `weight.n` selects which one a launcher runs.
inline constexpr std::int32_t kNvfp4AttnInputParentRows = Nvfp4N14336K5120::kOutputRows;
inline constexpr std::int32_t kNvfp4AttnInputShardRows  = Nvfp4N7168K5120::kOutputRows;

} // namespace ninfer::ops::detail
