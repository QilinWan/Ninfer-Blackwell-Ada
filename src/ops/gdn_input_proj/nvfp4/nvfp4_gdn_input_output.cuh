#pragma once

#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {

// The fused GDN Q|K|V|Z projection writes a Q|K|V plane and a trailing Z plane: the [16384,5120]
// parent covers 16 key heads (2048 rows paired with the query) and 48 value heads, so the sections
// are 2*2048+6144 = 10240 and 6144. Under two-device tensor parallelism each rank's standalone
// [8192,5120] shard stacks its own 8 of 16 key heads and 24 of 48 value heads in the same
// Q|K|V|Z order: 2*1024+3072 = 5120 and 3072. The shard keeps the parent's K, so its routes, their
// cutoffs and the workspace capacity are the parent's: only these section rows differ.
using Nvfp4GdnInputOutput      = LinearBf16SegmentedOutput<10240, 6144>;
using Nvfp4GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;

// Both problems are registered; `weight.n` selects which one a launcher runs.
inline constexpr std::int32_t kNvfp4GdnInputParentRows = Nvfp4N16384K5120::kOutputRows;
inline constexpr std::int32_t kNvfp4GdnInputShardRows  = Nvfp4N8192K5120::kOutputRows;

} // namespace ninfer::ops::detail
