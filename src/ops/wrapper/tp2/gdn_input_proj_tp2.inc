// Translation-unit fragment of src/ops/wrapper/gdn_input_proj.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) forms of gdn_input_proj, declared in ninfer/ops/tp2/gdn_input_proj.h.

#include "ninfer/ops/tp2/gdn_input_proj.h"
#include "ops/gdn_input_proj/tp2/fp8_gdn_input_shard.h"
#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"
#include "ops/common/split_launch.h"

#include <cstddef>

namespace ninfer::ops {

namespace {

// Two-device FP8 or NVFP4 shard profile; see the tensor-parallel section of gdn_input_proj.h.
constexpr std::int32_t kShardHidden    = 5120;
constexpr std::int32_t kShardKeyRows   = 1024;
constexpr std::int32_t kShardValueRows = 3072;
constexpr std::int32_t kShardChannels  = 2 * kShardKeyRows + kShardValueRows;
constexpr std::int32_t kShardRows      = kShardChannels + kShardValueRows;
static_assert(kShardRows == detail::Fp8N8192K5120::kOutputRows &&
              kShardRows == detail::Nvfp4N8192K5120::kOutputRows);
static_assert(kShardHidden == detail::Fp8N8192K5120::kInputRows &&
              kShardHidden == detail::Nvfp4N8192K5120::kInputRows);

void require_shard_profile(QType qtype, std::int32_t rows, std::int32_t input_rows,
                           const char* operation) {
    if ((qtype != QType::FP8_E4M3FN_ROW_BF16 && qtype != QType::NVFP4) || rows != kShardRows ||
        input_rows != kShardHidden) {
        throw std::invalid_argument(std::string(operation) + ": unsupported shard profile");
    }
}

void require_shard_weight(const Weight& weight, const char* operation) {
    if (weight.qtype == QType::NVFP4) {
        detail::validate_nvfp4_weight(weight, operation);
    } else if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        detail::validate_fp8_weight(weight, operation);
    } else {
        throw std::invalid_argument(std::string(operation) + ": unsupported weight format");
    }
    require_shard_profile(weight.qtype, weight.n, weight.k, operation);
}

// The bare projection's per-rank transient capacity over [min_tokens,max_tokens].
std::size_t shard_projection_bytes(QType qtype, LinearPolicy policy, std::int32_t min_tokens,
                                   std::int32_t max_tokens) {
    return qtype == QType::NVFP4
               ? detail::nvfp4_gdn_input_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                        max_tokens)
               : detail::fp8_gdn_input_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                      max_tokens);
}

void require_shard_workspace(const std::array<WorkspaceArena*, 2>& workspace, bool required,
                             const char* operation) {
    const std::size_t bytes = required ? 1U : 0U;
    detail::require_split_workspace(workspace, {bytes, bytes}, operation);
}

void require_shard_residency(const ExecutionContext& ec, int rank, const Tensor& x,
                             const Weight& weight, const Tensor& first, const Tensor& second,
                             const char* message) {
    detail::require_rank_residency(ec, rank, x.data, weight.payload, first.data, message);
    detail::require_rank_residency(ec, rank, nullptr, nullptr, second.data, message);
}

// The snapshot and record shards take the A8 or A4 frontier of the single-parent snapshot form,
// not the bare projection's, so that each rank quantizes exactly when the single-device Op does.
bool conv_shard_quantizes(QType qtype, LinearPolicy policy, ConvGeometry geometry) {
    return qtype == QType::NVFP4
               ? detail::nvfp4_gdn_conv_uses_a4(policy, geometry.width, geometry.batch)
               : detail::fp8_gdn_conv_uses_a8(policy, geometry.width, geometry.batch);
}

void project_conv_shard(const Tensor& x, const Weight& weight, Tensor& projected, Tensor& z,
                        LinearPolicy policy, ConvGeometry geometry, WorkspaceArena& workspace,
                        cudaStream_t stream) {
    const bool quantized = conv_shard_quantizes(weight.qtype, policy, geometry);
    if (weight.qtype == QType::NVFP4) {
        detail::nvfp4_gdn_input_shard_dispatch(
            x, weight, projected, z, quantized ? LinearPolicy::AllowA4 : LinearPolicy::A16Only,
            &workspace, stream);
    } else if (quantized) {
        detail::fp8_gdn_input_shard_a8_dispatch(x, weight, projected, z, workspace, stream);
    } else {
        detail::fp8_gdn_input_shard_a16_dispatch(x, weight, projected, z, stream);
    }
}

std::size_t conv_shard_projection_bytes(QType qtype, LinearPolicy policy, std::int32_t batch_size,
                                        std::int32_t max_width) {
    // The quantization frontier is monotonic in W, so the widest block bounds the interval.
    const std::int32_t columns = batch_size * max_width;
    if (!conv_shard_quantizes(qtype, policy, {max_width, batch_size, columns})) { return 0; }
    if (qtype == QType::NVFP4) {
        return detail::nvfp4_a4_workspace_capacity_bytes(columns, kShardHidden);
    }
    return detail::fp8_a8_workspace_capacity_bytes(
        columns, kShardHidden,
        detail::fp8_gdn_input_shard_partial_capacity_bytes(columns));
}

} // namespace

std::size_t gdn_input_proj_column_parallel_workspace_capacity_bytes(
    QType shard_qtype, std::int32_t shard_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel workspace: invalid token interval");
    }
    require_shard_profile(shard_qtype, shard_rows, input_rows,
                          "gdn_input_proj column-parallel workspace");
    return shard_projection_bytes(shard_qtype, policy, min_tokens, max_tokens);
}

void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_value_z_weight,
                                    const std::array<Tensor, 2>& qkv,
                                    const std::array<Tensor, 2>& z, LinearPolicy policy,
                                    const std::array<WorkspaceArena*, 2>& workspace,
                                    const ExecutionContext& ec) {
    constexpr const char* kOp = "gdn_input_proj column-parallel";
    validate_policy(policy);
    detail::require_split_pair(ec, x, query_key_value_z_weight, detail::SplitAxis::Output, kOp);
    const std::int32_t cols = x[0].ne[1];
    if (cols <= 0) {
        throw std::invalid_argument("gdn_input_proj column-parallel: T must be positive");
    }
    const QType qtype = query_key_value_z_weight[0].qtype;
    require_shard_workspace(workspace, shard_projection_bytes(qtype, policy, cols, cols) != 0, kOp);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        require_matrix(x[slot], kShardHidden, cols, "x");
        require_matrix(qkv[slot], kShardChannels, cols, "qkv");
        require_matrix(z[slot], kShardValueRows, cols, "z");
        require_single_parent_nonoverlap(x[slot], qkv[slot], z[slot]);
        require_shard_weight(query_key_value_z_weight[slot], kOp);
        require_shard_residency(ec, rank, x[slot], query_key_value_z_weight[slot], qkv[slot],
                                z[slot],
                                "gdn_input_proj column-parallel: rank arguments must reside on "
                                "its device");
    }
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        Tensor qkv_out  = qkv[slot];
        Tensor z_out    = z[slot];
        if (query_key_value_z_weight[slot].qtype == QType::NVFP4) {
            detail::nvfp4_gdn_input_shard_dispatch(x[slot], query_key_value_z_weight[slot],
                                                   qkv_out, z_out, policy, workspace[slot],
                                                   ec.dev[slot]->stream);
        } else {
            detail::fp8_gdn_input_shard_dispatch(x[slot], query_key_value_z_weight[slot], qkv_out,
                                                 z_out, policy, workspace[slot],
                                                 ec.dev[slot]->stream);
        }
    });
}

void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_value_z_weight,
                                    const std::array<Tensor, 2>& qkv,
                                    const std::array<Tensor, 2>& z, const ExecutionContext& ec) {
    gdn_input_proj_column_parallel(x, query_key_value_z_weight, qkv, z, LinearPolicy::A16Only,
                                   {nullptr, nullptr}, ec);
}

std::size_t gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
    QType shard_qtype, std::int32_t shard_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    require_shard_profile(shard_qtype, shard_rows, input_rows,
                          "gdn_input_proj_conv_snapshot column-parallel workspace");
    return composed_snapshot_capacity(
        kShardChannels, batch_size * max_width,
        conv_shard_projection_bytes(shard_qtype, policy, batch_size, max_width));
}

void gdn_input_proj_conv_snapshot_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_value_z_weight,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& snapshot_base_slots, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    constexpr const char* kOp = "gdn_input_proj_conv_snapshot column-parallel";
    validate_policy(policy);
    detail::require_split_pair(ec, x, query_key_value_z_weight, detail::SplitAxis::Output, kOp);
    require_shard_workspace(workspace, true, kOp);
    std::array<ConvGeometry, 2> geometry{};
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const Weight& w = query_key_value_z_weight[slot];
        geometry[slot]  = require_snapshot_input(x[slot], kShardHidden);
        require_shard_weight(w, kOp);
        require_snapshot_operands(conv_weight[slot], conv_states[slot], valid_columns[slot],
                                  initial_state_slots[slot], snapshot_base_slots[slot],
                                  kShardChannels, geometry[slot]);
        require_conv_tensor(query[slot], kShardKeyRows, geometry[slot].width, geometry[slot].batch,
                            kOp, "query");
        require_conv_tensor(key[slot], kShardKeyRows, geometry[slot].width, geometry[slot].batch,
                            kOp, "key");
        require_conv_tensor(value[slot], kShardValueRows, geometry[slot].width,
                            geometry[slot].batch, kOp, "value");
        require_conv_tensor(z[slot], kShardValueRows, geometry[slot].width, geometry[slot].batch,
                            kOp, "z");
        require_snapshot_nonoverlap(x[slot], conv_weight[slot], conv_states[slot],
                                    valid_columns[slot], initial_state_slots[slot],
                                    snapshot_base_slots[slot], query[slot], key[slot], value[slot],
                                    z[slot], *workspace[slot]);
        const std::array<const Tensor*, 10> tensors{&x[slot],
                                                    &conv_weight[slot],
                                                    &conv_states[slot],
                                                    &valid_columns[slot],
                                                    &initial_state_slots[slot],
                                                    &snapshot_base_slots[slot],
                                                    &query[slot],
                                                    &key[slot],
                                                    &value[slot],
                                                    &z[slot]};
        require_parent_nonoverlap(w, tensors, *workspace[slot], kOp);
        const char* residency =
            "gdn_input_proj_conv_snapshot column-parallel: rank arguments must reside on its "
            "device";
        require_shard_residency(ec, rank, x[slot], w, conv_states[slot], z[slot], residency);
        require_shard_residency(ec, rank, conv_weight[slot], w, query[slot], value[slot],
                                residency);
    }
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot     = static_cast<std::size_t>(rank);
        const Weight& w     = query_key_value_z_weight[slot];
        WorkspaceArena& ws  = *workspace[slot];
        cudaStream_t stream = ec.dev[slot]->stream;
        Tensor states       = conv_states[slot];
        Tensor query_out    = query[slot];
        Tensor key_out      = key[slot];
        Tensor value_out    = value[slot];
        Tensor z_out        = z[slot];
        compose_batched_snapshot(x[slot], conv_weight[slot], states, valid_columns[slot],
                                 initial_state_slots[slot], snapshot_base_slots[slot], query_out,
                                 key_out, value_out, z_out, kShardKeyRows, kShardKeyRows,
                                 kShardValueRows, geometry[slot], ws, stream,
                                 [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                     project_conv_shard(x_flat, w, projected, z_flat, policy,
                                                        geometry[slot], ws, stream);
                                 });
    });
}

std::size_t gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
    QType shard_qtype, std::int32_t shard_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_record_capacity_domain(batch_size, min_width, max_width);
    require_shard_profile(shard_qtype, shard_rows, input_rows,
                          "gdn_input_proj_conv_record column-parallel workspace");
    return conv_shard_projection_bytes(shard_qtype, policy, batch_size, max_width);
}

void gdn_input_proj_conv_record_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_value_z_weight,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& conv_record, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    constexpr const char* kOp = "gdn_input_proj_conv_record column-parallel";
    validate_policy(policy);
    detail::require_split_pair(ec, x, query_key_value_z_weight, detail::SplitAxis::Output, kOp);
    require_shard_workspace(workspace, true, kOp);
    std::array<ConvGeometry, 2> geometry{};
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const Weight& w = query_key_value_z_weight[slot];
        geometry[slot]  = require_record_input(x[slot], kShardHidden);
        require_shard_weight(w, kOp);
        require_record_operands(conv_weight[slot], conv_states[slot], valid_columns[slot],
                                initial_state_slots[slot], kShardChannels, geometry[slot]);
        require_conv_tensor(conv_record[slot], kShardChannels, geometry[slot].width,
                            geometry[slot].batch, kOp, "conv record");
        require_conv_tensor(query[slot], kShardKeyRows, geometry[slot].width, geometry[slot].batch,
                            kOp, "query");
        require_conv_tensor(key[slot], kShardKeyRows, geometry[slot].width, geometry[slot].batch,
                            kOp, "key");
        require_conv_tensor(value[slot], kShardValueRows, geometry[slot].width,
                            geometry[slot].batch, kOp, "value");
        require_conv_tensor(z[slot], kShardValueRows, geometry[slot].width, geometry[slot].batch,
                            kOp, "z");
        require_record_nonoverlap(x[slot], conv_weight[slot], conv_states[slot],
                                  valid_columns[slot], initial_state_slots[slot], conv_record[slot],
                                  query[slot], key[slot], value[slot], z[slot], *workspace[slot]);
        const std::array<const Tensor*, 10> tensors{&x[slot],
                                                    &conv_weight[slot],
                                                    &conv_states[slot],
                                                    &valid_columns[slot],
                                                    &initial_state_slots[slot],
                                                    &conv_record[slot],
                                                    &query[slot],
                                                    &key[slot],
                                                    &value[slot],
                                                    &z[slot]};
        require_parent_nonoverlap(w, tensors, *workspace[slot], kOp);
        const char* residency =
            "gdn_input_proj_conv_record column-parallel: rank arguments must reside on its device";
        require_shard_residency(ec, rank, x[slot], w, conv_record[slot], z[slot], residency);
        require_shard_residency(ec, rank, conv_states[slot], w, query[slot], value[slot],
                                residency);
    }
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot     = static_cast<std::size_t>(rank);
        const Weight& w     = query_key_value_z_weight[slot];
        WorkspaceArena& ws  = *workspace[slot];
        cudaStream_t stream = ec.dev[slot]->stream;
        Tensor record_out   = conv_record[slot];
        Tensor query_out    = query[slot];
        Tensor key_out      = key[slot];
        Tensor value_out    = value[slot];
        Tensor z_out        = z[slot];
        compose_record(x[slot], conv_weight[slot], conv_states[slot], valid_columns[slot],
                       initial_state_slots[slot], record_out, query_out, key_out, value_out, z_out,
                       geometry[slot], ws, stream,
                       [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                           project_conv_shard(x_flat, w, record_flat, z_flat, policy,
                                              geometry[slot], ws, stream);
                       });
    });
}

} // namespace ninfer::ops
