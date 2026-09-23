#include "models/qwen3_5/execution/mtp.h"

#include "core/layout.h"
#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/mtp_pack.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::execution {

std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                           std::int32_t first, std::int32_t last) {
    const auto& p = parameters.packed;
    const auto& w = p.weight;
    if (!parameters.rows) {
        return ops::attn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first,
                                                             last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                   const AttentionConfig& config, std::int32_t first,
                                   std::int32_t last) {
    if (parameters.rows) {
        return ops::linear_pair_workspace_capacity_bytes((*parameters.rows)[1].weight,
                                                         (*parameters.rows)[3].weight, first, last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                           const AttentionConfig& config, std::int32_t first,
                                           std::int32_t last) {
    if (parameters.rows) {
        const auto bytes = [&](std::size_t index) {
            const auto& p = (*parameters.rows)[index];
            const auto& w = p.weight;
            return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
        };
        return std::max(bytes(0), bytes(2));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& p = parameters.packed;
    if (!parameters.rows) {
        ops::attn_input_proj(hidden, p.weight, query, gate, key, value, p.policy, workspace,
                             stream);
        return;
    }
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    Tensor packed      = workspace.alloc(DType::BF16, {p.weight.n, columns});
    ops::linear(hidden, p.weight, packed, p.policy, workspace, stream);
    Tensor q =
        query.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor k =
        key.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    Tensor g =
        gate.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor v =
        value.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    ops::mtp_split_attn_in(packed, q, k, g, v, stream);
}

void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        ops::linear_pair(hidden, (*parameters.rows)[1].weight, (*parameters.rows)[3].weight, key,
                         value, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    Tensor gate  = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        const auto& q = (*parameters.rows)[0];
        const auto& g = (*parameters.rows)[2];
        {
            auto scope = workspace.scope();
            ops::linear(hidden, q.weight, query, q.policy, workspace, stream);
        }
        ops::linear(hidden, g.weight, gate, g.policy, workspace, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor key   = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    Tensor value = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

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
    int previous        = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    for (std::size_t r = 0; r < 2; ++r) {
        CUDA_CHECK(cudaSetDevice(execution.dev[r]->device));
        Tensor q = query[r].view({head_dim, q_heads, columns});
        Tensor k = key[r].view({head_dim, kv_heads, columns});
        Tensor g = gate[r].view({head_dim, q_heads, columns});
        Tensor v = value[r].view({head_dim, kv_heads, columns});
        ops::mtp_split_attn_in(packed[r], q, k, g, v, execution.dev[r]->stream);
    }
    CUDA_CHECK(cudaSetDevice(previous));
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
