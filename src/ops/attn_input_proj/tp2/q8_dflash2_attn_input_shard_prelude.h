#pragma once

// Included first by the Q8 DFlash2 shard translation unit of this directory: the family's headers,
// processed once under their own names before q8_dflash2_attn_input_shard_names.h renames
// anything.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"
#include "ops/attn_input_proj/tp2/q8_dflash2_attn_input_shard.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

namespace ninfer::ops::detail {

// The shard's query|key|value section output: a rank's 16 of the 32 query heads and 4 of the 8
// KV heads of D128, in the parent's section order. The parent's segment sizes, which the upstream
// source writes as template arguments, are ignored.
template <int, int, int>
using Q8DFlash2AttnInputShardOutput =
    LinearBf16SegmentedOutput<kQ8DFlash2ShardQueryRows, kQ8DFlash2ShardKvRows,
                              kQ8DFlash2ShardKvRows>;

} // namespace ninfer::ops::detail
