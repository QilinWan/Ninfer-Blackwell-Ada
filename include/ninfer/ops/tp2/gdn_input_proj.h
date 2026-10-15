#pragma once

// Two-device (tp 2) forms of gdn_input_proj. Our file: ninfer/ops/gdn_input_proj.h is upstream's and carries no hook.

#include "core/device.h"
#include "ninfer/ops/gdn_input_proj.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Tensor-parallel forms over two devices.
//
// Rank r owns key heads [8r,8r+8) and value heads [24r,24r+24). Its weight is a standalone
// FP8_E4M3FN_ROW_BF16 RowScale or NVFP4 BlockScaleK16M128x4 [8192,5120] parent concatenating, in
// the parent's Q|K|V|Z order, its rows of every section:
//
//   Q rows [1024r,+1024) | K [2048+1024r,+1024) | V [4096+3072r,+3072) | Z [10240+3072r,+3072)
//
// so the shard's sections sit at [0,1024), [1024,2048), [2048,5120) and [5120,8192). Its
// convolution channels are the matching 1024+1024+3072 channels of the 10240-channel parent in the
// same order, and a shard-local head index plus 8r (Q/K) or 24r (V/Z) is the global head. `x`
// holds the same activation on both ranks. Convolution is depthwise and the projection is
// column-parallel, so nothing is communicated and each rank's results match the corresponding
// sections of the single-device Op to rounding: the shard always materializes the projection
// before the convolution, where the single-device form may fuse the two. Other formats are not
// registered.
//
// Every requirement of the single-device form of the same format applies per rank at the shard
// profile, and each form's activation-quantization frontier is that of the same form over the
// [16384,5120] parent (FP8 snapshot and record forms use A8 from W=10 at B=1 and from B*W=9 when
// batched; NVFP4 ones use A4 under AllowA4 from W=4 at B=1 and at every batched W). Rank
// r's tensors, weight and workspace must be resident on `ec.dev[r]`, and its work is enqueued on
// `ec.dev[r]->stream`. Inputs staged on a device's legacy default stream must be retired before
// the call; the call does not synchronize and preserves the current device.

/// Per-rank transient capacity of gdn_input_proj_column_parallel for a shard profile.
[[nodiscard]] std::size_t gdn_input_proj_column_parallel_workspace_capacity_bytes(
    QType shard_qtype, std::int32_t shard_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Column-parallel gdn_input_proj. Per rank: x [5120,T], qkv [5120,T] (Q|K|V) and z [3072,T].
 */
void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_value_z_weight,
                                    const std::array<Tensor, 2>& qkv,
                                    const std::array<Tensor, 2>& z, LinearPolicy policy,
                                    const std::array<WorkspaceArena*, 2>& workspace,
                                    const ExecutionContext& ec);

/// A16-only column-parallel form; it requires no transient workspace.
/// Model execution passes a policy; this form is the A16 entry the op qualification suites use.
void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_value_z_weight,
                                    const std::array<Tensor, 2>& qkv,
                                    const std::array<Tensor, 2>& z, const ExecutionContext& ec);

/// Per-rank transient capacity of gdn_input_proj_conv_snapshot_column_parallel, over the same B/W
/// domain as the single-parent query.
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
    QType shard_qtype, std::int32_t shard_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Column-parallel gdn_input_proj_conv_snapshot. Per rank: x [5120,W,B], conv_weight [5120,4],
 * conv_states [5120,3,Slots], query/key [1024,W,B], value/z [3072,W,B], and the I32 selectors of
 * the single-device form. The B/W domain, state contract and non-overlap rules are those of the
 * single-parent form. Each rank projects its shard into a BF16 [5120,W*B] workspace plane and then
 * applies the convolution, so the shard does not use the fused FP8 or NVFP4 snapshot kernels.
 */
void gdn_input_proj_conv_snapshot_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_value_z_weight,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& snapshot_base_slots, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);

/// Per-rank transient capacity of gdn_input_proj_conv_record_column_parallel.
[[nodiscard]] std::size_t gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
    QType shard_qtype, std::int32_t shard_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Column-parallel gdn_input_proj_conv_record, with the shard operands of the snapshot form and
 * conv_record [5120,T,B]. The projection is written directly to conv_record. Outputs and records
 * equal the corresponding snapshot form's from the same inputs.
 */
void gdn_input_proj_conv_record_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_value_z_weight,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& conv_record, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);

} // namespace ninfer::ops
