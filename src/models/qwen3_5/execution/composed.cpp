#include "models/qwen3_5/execution/composed.h"

#include "core/device.h"
#include "core/dtype.h"
#include "core/layout.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::int32_t columns_for(const Tensor& t, std::int32_t rows) {
    const std::int64_t numel = t.numel();
    if (rows <= 0 || numel <= 0 || numel % rows != 0) {
        throw std::invalid_argument("composed projection: operand does not tile its row extent");
    }
    return static_cast<std::int32_t>(numel / rows);
}

std::size_t linear_scratch_bytes(const LinearParameters& p, std::int32_t first,
                                 std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                                first, last);
}

} // namespace

bool rows_share_one_parent(std::span<const ops::WeightInput> inputs) {
    const WeightParent* parent = nullptr;
    std::uint64_t end          = 0;
    for (const ops::WeightInput& input : inputs) {
        if (input.weight.parts.empty()) { return false; }
        for (const WeightRegion& part : input.weight.parts) {
            if (part.parent == nullptr || part.end <= part.begin) { return false; }
            if (parent == nullptr) {
                parent = part.parent;
            } else if (part.parent != parent || part.begin != end) {
                return false;
            }
            end = part.end;
        }
    }
    return parent != nullptr;
}

// The dense groupwise catalogs register exactly the Qwen3.6/3.8-27B widths; the MoE widths run
// through Q8 single parents. Ask the plans rather than repeat their tables here.
bool registered_attention_projection(const TextConfig& config) {
    if (!config.attention) { return false; }
    const auto& a = *config.attention;
    return ops::detail::q4_q5_attn_input_admits(
        {dimension(config.hidden_size), dimension(a.query_width()), dimension(a.key_width()),
         dimension(config.hidden_size), 1});
}

bool registered_gdn_projection(const TextConfig& config) {
    if (!config.gdn) { return false; }
    const auto& g       = *config.gdn;
    const auto key      = dimension(g.key_width());
    const auto value    = dimension(g.value_width());
    const auto hidden   = dimension(config.hidden_size);
    return ops::detail::q4_q5_gdn_input_admits(
        {hidden, 2 * key, 2 * value, 2 * key + value, value, hidden, 1});
}

bool registered_gdn_control(const TextConfig& config) {
    if (!config.gdn) { return false; }
    return ops::detail::bf16_gdn_gating_admits(
        {dimension(config.gdn->linear_num_value_heads), dimension(config.hidden_size), 1});
}

bool registered_swiglu(std::int32_t gate_up_rows, std::int32_t k) {
    return ops::detail::q4_linear_swiglu_admits({gate_up_rows, gate_up_rows / 2, k, k, 1});
}

bool registered_linear_add(std::int32_t rows, std::int32_t k) {
    return ops::detail::q5_linear_add_admits({rows, k, k, 1});
}

bool registered_pair(std::int32_t rows, std::int32_t k) {
    return ops::detail::q8_pair_admits({rows, k, k, 1});
}

bool registered_mtp_split(const AttentionConfig& config) {
    return config.head_dim == 256 && config.num_attention_heads == 24 &&
           config.num_key_value_heads == 4;
}

Tensor as_matrix(const Tensor& t, std::int32_t rows) {
    if (!t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument("composed projection: operand must be contiguous");
    }
    return Tensor(t.data, t.dtype, {rows, columns_for(t, rows)});
}

void composed_linear(const Tensor& x, const LinearParameters& p, Tensor& out,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor x_matrix   = as_matrix(x, p.weight.k);
    Tensor out_matrix = as_matrix(out, p.weight.n);
    auto scope        = workspace.scope();
    ops::linear(x_matrix, p.weight, out_matrix, p.policy, workspace, stream);
}

// The linear op writes its result with the leading dimension of the projection, so a channel band
// of a wider matrix has to be staged and copied in: the two parents are different quantized
// formats, so the split form cannot be a single weight either.
void copy_channel_band(const Tensor& source, Tensor& destination, std::int32_t destination_channel0,
                       std::int32_t channels, std::int32_t columns, cudaStream_t stream) {
    const std::size_t element_bytes = dtype_size(source.dtype);
    auto* target = static_cast<std::byte*>(destination.data) +
                   static_cast<std::size_t>(destination_channel0) * element_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, static_cast<std::size_t>(destination.nb[1]), source.data,
                                 static_cast<std::size_t>(source.nb[1]),
                                 static_cast<std::size_t>(channels) * element_bytes,
                                 static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                                 stream));
}

// The query|key|value logical matrix, whether the artifact keeps it in one parent or (as the
// ninfer-5080 groupwise converter does) splits `query_key` from `value_z`.
void project_gdn_rows(const Tensor& hidden, const ComposedGdnProjection& p, Tensor& rows,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    if (!p.split) {
        composed_linear(hidden, p.query_key_value, rows, workspace, stream);
        return;
    }
    const std::int32_t columns   = columns_for(rows, p.channels);
    const std::int32_t query_key = p.query_key_value.weight.n;
    auto scope                   = workspace.scope();
    Tensor staged_query_key      = workspace.alloc(DType::BF16, {query_key, columns});
    Tensor staged_value          = workspace.alloc(DType::BF16, {p.value.weight.n, columns});
    composed_linear(hidden, p.query_key_value, staged_query_key, workspace, stream);
    composed_linear(hidden, p.value, staged_value, workspace, stream);
    copy_channel_band(staged_query_key, rows, 0, query_key, columns, stream);
    copy_channel_band(staged_value, rows, query_key, p.value.weight.n, columns, stream);
}

void composed_linear_add(const Tensor& x, const LinearParameters& p, Tensor& residual,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope   = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {p.weight.n, columns_for(x, p.weight.k)});
    composed_linear(x, p, delta, workspace, stream);
    Tensor residual_matrix = as_matrix(residual, p.weight.n);
    ops::residual_add(delta, residual_matrix, stream);
}

std::size_t composed_linear_add_workspace_bytes(const LinearParameters& p, std::int32_t first,
                                                std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {p.weight.n, last});
    (void)layout.alloc_bytes(linear_scratch_bytes(p, first, last));
    return layout.peak_bytes(1);
}

void composed_attention_projection(const Tensor& hidden, const ComposedAttentionProjection& p,
                                   Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    composed_linear(hidden, p.query, query, workspace, stream);
    composed_linear(hidden, p.key, key, workspace, stream);
    composed_linear(hidden, p.gate, gate, workspace, stream);
    composed_linear(hidden, p.value, value, workspace, stream);
}

void composed_gdn_projection(const Tensor& hidden, const ComposedGdnProjection& p, Tensor& qkv,
                             Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    project_gdn_rows(hidden, p, qkv, workspace, stream);
    composed_linear(hidden, p.z, z, workspace, stream);
}

std::size_t composed_gdn_projection_workspace_bytes(const ComposedGdnProjection& p,
                                                    std::int32_t first, std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(linear_scratch_bytes(p.query_key_value, first, last));
    (void)layout.alloc_bytes(linear_scratch_bytes(p.z, first, last));
    if (p.split) {
        (void)layout.alloc(DType::BF16, {p.query_key_value.weight.n, last});
        (void)layout.alloc(DType::BF16, {p.value.weight.n, last});
        (void)layout.alloc_bytes(linear_scratch_bytes(p.value, first, last));
    }
    return layout.peak_bytes(1);
}

std::size_t composed_gdn_snapshot_workspace_bytes(const ComposedGdnProjection& p,
                                                  std::int32_t batch, std::int32_t first_width,
                                                  std::int32_t last_width) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {p.channels, batch * last_width});
    (void)layout.alloc_bytes(linear_scratch_bytes(p.query_key_value, batch * first_width,
                                                  batch * last_width));
    (void)layout.alloc_bytes(linear_scratch_bytes(p.z, batch * first_width,
                                                  batch * last_width));
    if (p.split) {
        (void)layout.alloc(DType::BF16, {p.query_key_value.weight.n, batch * last_width});
        (void)layout.alloc(DType::BF16, {p.value.weight.n, batch * last_width});
        (void)layout.alloc_bytes(linear_scratch_bytes(p.value, batch * first_width,
                                                      batch * last_width));
    }
    return layout.peak_bytes(1);
}

void composed_gdn_snapshot(const Tensor& hidden, const ComposedGdnProjection& p,
                           const Tensor& convolution, Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           const Tensor& destination_slots, Tensor& query, Tensor& key,
                           Tensor& value, Tensor& z, WorkspaceArena& workspace,
                           cudaStream_t stream) {
    // hidden is [hidden, width, batch]; project the width*batch columns in one pass, then let the
    // family launcher run the causal convolution over the [channels, width, batch] result.
    const std::int32_t width    = hidden.ne[1];
    const std::int32_t batch    = hidden.ne[2];
    const std::int32_t channels = p.channels;
    auto scope                  = workspace.scope();
    Tensor projected            = workspace.alloc(DType::BF16, {channels, width * batch});
    project_gdn_rows(hidden, p, projected, workspace, stream);
    composed_linear(hidden, p.z, z, workspace, stream);
    Tensor projected_view(projected.data, DType::BF16, {channels, width, batch});
    ops::detail::gdn_projected_conv_snapshot_launch(projected_view, convolution, conv_states,
                                                    valid_columns, initial_slots,
                                                    destination_slots, query, key, value, stream);
}

void composed_gdn_record(const Tensor& hidden, const ComposedGdnProjection& p,
                         const Tensor& convolution, const Tensor& conv_states,
                         const Tensor& valid_columns, const Tensor& initial_slots,
                         Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    // The record row is the projection itself; the launcher reads it in record mode and leaves
    // the persistent state untouched.
    project_gdn_rows(hidden, p, conv_record, workspace, stream);
    composed_linear(hidden, p.z, z, workspace, stream);
    ops::detail::gdn_projected_conv_record_launch(conv_record, convolution, conv_states,
                                                  valid_columns, initial_slots, query, key, value,
                                                  stream);
}

std::size_t composed_gdn_control_workspace_bytes(const ComposedGdnControl& p, std::int32_t first,
                                                 std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {p.a.weight.n, last});
    (void)layout.alloc(DType::BF16, {p.b.weight.n, last});
    (void)layout.alloc_bytes(linear_scratch_bytes(p.a, first, last));
    return layout.peak_bytes(1);
}

void composed_gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                               const ComposedGdnControl& p, const Tensor& a_log,
                               const Tensor& dt_bias, Tensor& hidden, Tensor& g, Tensor& beta,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = columns_for(residual, p.a.weight.k);
    ops::rmsnorm(residual, norm, epsilon, true, hidden, stream);
    Tensor a = workspace.alloc(DType::BF16, {p.a.weight.n, cols});
    Tensor b = workspace.alloc(DType::BF16, {p.b.weight.n, cols});
    composed_linear(hidden, p.a, a, workspace, stream);
    composed_linear(hidden, p.b, b, workspace, stream);
    ops::gdn_gating(a, b, a_log, dt_bias, g, beta, stream);
}

} // namespace ninfer::models::qwen3_5::execution
