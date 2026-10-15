#pragma once

// Included first by every FP8 shard translation unit of this directory: the family's headers,
// processed once under their own names before fp8_attn_input_shard_names.h renames anything.

#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/tp2/fp8_attn_input_shard.h"
#include "ops/linear/fp8/fp8_geometry.h"

namespace ninfer::ops::detail {

// The shard's query|key|gate|value section output, in the parent's section order.
using Fp8AttnInputShardOutput = LinearBf16SegmentedOutput<3072, 512, 3072, 512>;

static_assert(Fp8N7168K5120::kOutputRows == 2 * (3072 + 512));
static_assert(Fp8N7168K5120::kInputRows == Fp8N14336K5120::kInputRows);

} // namespace ninfer::ops::detail
