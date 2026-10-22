#include "models/qwen3_5/execution/attention.h"

#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rope.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

const LinearParameters& split_projection(const AttentionParameters& parameters) {
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    if (single == nullptr) {
        throw std::invalid_argument("tensor-parallel attention projection: a paired Q|K + gate|V "
                                    "projection has no two-device route");
    }
    return *single;
}

} // namespace

std::size_t attention_projection_split_workspace_bytes(const AttentionParameters& parameters,
                                                       std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    const auto& single = split_projection(parameters);
    return ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
        single.weight.qtype, single.policy, first, last);
}

void attention_projection_split(const std::array<Tensor, 2>& hidden,
                                const std::array<const AttentionParameters*, 2>& parameters,
                                const std::array<Tensor, 2>& query,
                                const std::array<Tensor, 2>& gate, const std::array<Tensor, 2>& key,
                                const std::array<Tensor, 2>& value,
                                const std::array<WorkspaceArena*, 2>& workspace,
                                const ExecutionContext& execution) {
    const std::array<const LinearParameters*, 2> shards{&split_projection(*parameters[0]),
                                                        &split_projection(*parameters[1])};
    const auto policy = split_policy(shards, "tensor-parallel attention projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    ops::attn_input_proj_column_parallel(hidden, split_weights(shards), query, gate, key, value,
                                         policy, workspace, execution);
}

void attention_output_split(const std::array<Tensor, 2>& attention,
                            const std::array<const AttentionParameters*, 2>& parameters,
                            const std::array<Tensor, 2>& residual,
                            const std::array<Tensor, 2>& staging,
                            const std::array<WorkspaceArena*, 2>& workspace,
                            const ExecutionContext& execution, const ops::PeerEvents& events,
                            int exchange_lane) {
    project_add_row_parallel(attention, {&parameters[0]->output, &parameters[1]->output}, residual,
                             staging, workspace, execution, events, exchange_lane);
}

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                             single->policy, first, last);
    }
    return 0;
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(hidden, single.weight, query, gate, key, value, single.policy,
                             workspace, stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config,
               const ops::TextRopeScaling* scaling, Tensor& query, DeviceExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, execution,
              scaling);
}

void text_rope(const Tensor& positions, const RopeConfig& config,
               const ops::TextRopeScaling* scaling, Tensor& query, Tensor& key,
               DeviceExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, execution,
              scaling);
}

} // namespace ninfer::models::qwen3_5::execution
