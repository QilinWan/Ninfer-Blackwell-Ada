#pragma once

#include "artifact/binder.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/load/bindings.h"

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::loading {

// Tensor-parallel placement of one logical parameter, in the parameter's own coordinates:
// `ranges[d]` are the rows (Rows) or columns (Columns) device d computes with. The artifact
// objects stay opaque; parent_placements() maps these ranges through each Binding part.
//
// Dense Qwen3.5 at tp > 1 (heads always split into contiguous per-rank blocks, which keeps every
// GDN value head with its key head and every query head with its KV head):
//
//   Rows     attention/{query,gate} by query heads, attention/{key,value} by KV heads,
//            gdn/{query,key} by key heads, gdn/{value,z,a_projection,b_projection,a_log,dt_bias}
//            by value heads, mlp/{gate,up} by intermediate rows, text/output_head by vocabulary.
//   Columns  attention/output, gdn/output, mlp/down and mtp/input_projection over their input
//            (rank 0 contracts the normalized embedding half of mtp/input_projection, rank 1 the
//            hidden half); gdn/convolution over its Q|K|V channel sections, one range each.
//   Replicated  norms, text/token_embedding.
//   SingleDevice(vision_rank)  vision/*.
//   PrimaryOnly  dflash/*, dflash2/* and proposal/*: the drafter and the optimized proposal head
//            run on rank 0 only, so rank 1 holds none of them.
//
// MTP and a masked draft share text/token_embedding and the selected output head by WeightId,
// so they see that parameter's per-rank view: MTP runs on both ranks and gathers the
// vocabulary-split text/output_head; the rank-0 drafter requires the whole optimized proposal
// head (plan_load rejects a masked draft with the full head at tp > 1).
struct LogicalShard {
    artifact::ShardAxis axis = artifact::ShardAxis::Replicated;
    int device               = 0; // Holder under SingleDevice.
    std::array<std::vector<artifact::SliceRange>, artifact::kMaximumDevices> ranges;
};

// Throws std::invalid_argument unless options.tp is in [1, 2] and options.vision_rank names one
// of its ranks. plan_load() checks this once before binding; logical_shard() and
// parent_placements() check it again for their direct callers.
void validate_tensor_parallel_ranks(const LoadOptions& options);

// Throws std::invalid_argument for a parameter without a tensor-parallel rule, for a head or
// width that does not divide by options.tp, and for options validate_tensor_parallel_ranks()
// rejects.
[[nodiscard]] LogicalShard logical_shard(std::string_view name, const artifact::Shape& shape,
                                         const Config& config, const LoadOptions& options);

using GeometryLookup = std::function<const WeightGeometry&(artifact::ObjectHandle)>;

// Combines the logical placements of every parameter bound to one parent. Row ranges map
// through the Binding parts to whole parent rows and must partition the parent; column ranges
// must agree across parameters and partition its columns; complete-parent kinds must agree.
// Indexed by object; parents that no parameter demands on a device stay empty.
[[nodiscard]] std::vector<std::optional<artifact::ShardPlacement>>
parent_placements(std::span<const PendingWeight> weights, std::size_t object_count,
                  const GeometryLookup& geometry, const Config& config, const LoadOptions& options);

// Installs the combined placements on a Binder planned for options.tp devices. No-op at tp 1.
void install_shard_resolver(artifact::Binder& binder, std::span<const PendingWeight> weights,
                            const Config& config, const LoadOptions& options);

} // namespace ninfer::models::qwen3_5::loading
