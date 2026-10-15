#pragma once

// Included first by every FP8 shard translation unit of this directory: the family's headers,
// processed once under their own names before fp8_gdn_input_shard_names.h renames anything.

#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/tp2/fp8_gdn_input_shard.h"
#include "ops/linear/fp8/fp8_geometry.h"

namespace ninfer::ops::detail {

// The shard's qkv|z section output.
using Fp8GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;

static_assert(Fp8N8192K5120::kOutputRows == 5120 + 3072);
static_assert(Fp8N8192K5120::kInputRows == Fp8N16384K5120::kInputRows);

} // namespace ninfer::ops::detail
