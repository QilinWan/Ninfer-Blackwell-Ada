#include "models/qwen3_5/execution/tp2/mtp_split.h"

#include "core/device_scope.h"
#include "core/layout.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/mtp_pack.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

std::size_t mtp_projection_split_workspace_bytes(const MtpProjectionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    const auto& p = parameters.packed;
    const auto& w = p.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_split_workspace_bytes(const MtpProjectionParameters& parameters,
                                         const AttentionConfig& shard, std::int32_t first,
                                         std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(shard.query_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(shard.query_width()), last});
    (void)layout.alloc_bytes(mtp_projection_split_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_query_gate_split_workspace_bytes(const MtpProjectionParameters& parameters,
                                                 const AttentionConfig& shard, std::int32_t first,
                                                 std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(shard.key_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(shard.key_width()), last});
    (void)layout.alloc_bytes(mtp_projection_split_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

void mtp_projection_split(const std::array<Tensor, 2>& hidden,
                          const std::array<const MtpProjectionParameters*, 2>& parameters,
                          const AttentionConfig& shard, const std::array<Tensor, 2>& query,
                          const std::array<Tensor, 2>& gate, const std::array<Tensor, 2>& key,
                          const std::array<Tensor, 2>& value,
                          const std::array<WorkspaceArena*, 2>& workspace,
                          const ExecutionContext& execution) {
    auto scope0        = workspace[0]->scope();
    auto scope1        = workspace[1]->scope();
    const auto columns = hidden[0].ne[1];
    std::array<Tensor, 2> packed;
    for (std::size_t r = 0; r < 2; ++r) {
        packed[r] = workspace[r]->alloc(DType::BF16, {parameters[r]->packed.weight.n, columns});
    }
    project_column_parallel(hidden, {&parameters[0]->packed, &parameters[1]->packed}, packed,
                            workspace, execution);
    const auto head_dim = dimension(shard.head_dim);
    const auto q_heads  = dimension(shard.num_attention_heads);
    const auto kv_heads = dimension(shard.num_key_value_heads);
    const ScopedCurrentDevice restore;
    for (std::size_t r = 0; r < 2; ++r) {
        ScopedCurrentDevice::select(execution.dev[r]->device);
        Tensor q = query[r].view({head_dim, q_heads, columns});
        Tensor k = key[r].view({head_dim, kv_heads, columns});
        Tensor g = gate[r].view({head_dim, q_heads, columns});
        Tensor v = value[r].view({head_dim, kv_heads, columns});
        ops::mtp_split_attn_in(packed[r], q, k, g, v, execution.dev[r]->stream);
    }
}

void mtp_kv_projection_split(const std::array<Tensor, 2>& hidden,
                             const std::array<const MtpProjectionParameters*, 2>& parameters,
                             const AttentionConfig& shard, const std::array<Tensor, 2>& key,
                             const std::array<Tensor, 2>& value,
                             const std::array<WorkspaceArena*, 2>& workspace,
                             const ExecutionContext& execution) {
    auto scope0        = workspace[0]->scope();
    auto scope1        = workspace[1]->scope();
    const auto columns = hidden[0].ne[1];
    std::array<Tensor, 2> query;
    std::array<Tensor, 2> gate;
    for (std::size_t r = 0; r < 2; ++r) {
        query[r] = workspace[r]->alloc(DType::BF16, {dimension(shard.query_width()), columns});
        gate[r]  = workspace[r]->alloc(DType::BF16, {dimension(shard.query_width()), columns});
    }
    mtp_projection_split(hidden, parameters, shard, query, gate, key, value, workspace, execution);
}

void mtp_query_gate_projection_split(
    const std::array<Tensor, 2>& hidden,
    const std::array<const MtpProjectionParameters*, 2>& parameters, const AttentionConfig& shard,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& gate,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution) {
    auto scope0        = workspace[0]->scope();
    auto scope1        = workspace[1]->scope();
    const auto columns = hidden[0].ne[1];
    std::array<Tensor, 2> key;
    std::array<Tensor, 2> value;
    for (std::size_t r = 0; r < 2; ++r) {
        key[r]   = workspace[r]->alloc(DType::BF16, {dimension(shard.key_width()), columns});
        value[r] = workspace[r]->alloc(DType::BF16, {dimension(shard.key_width()), columns});
    }
    mtp_projection_split(hidden, parameters, shard, query, gate, key, value, workspace, execution);
}

void mtp_input_projection_split(const std::array<Tensor, 2>& input,
                                const std::array<const MtpParameters*, 2>& parameters,
                                const std::array<Tensor, 2>& output,
                                const std::array<Tensor, 2>& staging,
                                const std::array<WorkspaceArena*, 2>& workspace,
                                const ExecutionContext& execution, const ops::PeerEvents& events) {
    project_row_parallel(input,
                         {&parameters[0]->input_projection, &parameters[1]->input_projection},
                         output, staging, workspace, execution, events);
}

void mtp_output_split(const std::array<Tensor, 2>& attention,
                      const std::array<const MtpParameters*, 2>& parameters,
                      const std::array<Tensor, 2>& output, const std::array<Tensor, 2>& staging,
                      const std::array<WorkspaceArena*, 2>& workspace,
                      const ExecutionContext& execution, const ops::PeerEvents& events) {
    project_row_parallel(attention, {&parameters[0]->output, &parameters[1]->output}, output,
                         staging, workspace, execution, events);
}

} // namespace ninfer::models::qwen3_5::execution
