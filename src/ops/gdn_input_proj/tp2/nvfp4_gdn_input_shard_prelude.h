#pragma once

// Included first by every NVFP4 shard translation unit of this directory: the family's headers,
// processed once under their own names before nvfp4_gdn_input_shard_names.h renames anything.

#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {

// The shard's qkv|z section output. Both sections are whole 128-row tiles, like the parent's.
using Nvfp4GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;

static_assert(Nvfp4N8192K5120::kOutputRows == 5120 + 3072);
static_assert(Nvfp4N8192K5120::kInputRows == Nvfp4N16384K5120::kInputRows);
static_assert((5120 % 128) == 0 && (3072 % 128) == 0);

} // namespace ninfer::ops::detail
