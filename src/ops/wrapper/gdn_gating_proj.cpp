#include "core/weight.h"
#include "ninfer/ops/gdn_gating_proj.h"

#include "ops/common/split_launch.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_bf16_weight(const Weight& w, std::int32_t rows, std::int32_t input_rows,
                         const char* name) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(input_rows) *
                                        sizeof(std::uint16_t);
    if (w.qtype != QType::BF16 || w.layout != QuantLayout::Contiguous ||
        w.payload_bytes < payload_bytes || w.ndim != 2 || w.n != rows || w.k != input_rows ||
        w.shape[0] != rows || w.shape[1] != input_rows || w.padded_shape[0] != rows ||
        w.padded_shape[1] != input_rows || w.qhigh != nullptr || w.scales != nullptr ||
        w.high_plane_bytes != 0 || w.group != 0 || w.group_size != 0 || !aligned_to(w.qdata, 16)) {
        throw std::invalid_argument(std::string("gdn_gating_proj: invalid ") + name);
    }
}

Weight bf16_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    const std::size_t row_bytes = static_cast<std::size_t>(parent.k) * sizeof(std::uint16_t);
    const auto* data            = static_cast<const std::uint8_t*>(parent.qdata) +
                       static_cast<std::size_t>(row_begin) * row_bytes;
    Weight view          = parent;
    view.payload         = data;
    view.payload_bytes   = static_cast<std::uint64_t>(rows) * row_bytes;
    view.qdata           = data;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    view.n               = rows;
    return view;
}

struct GdnControlParentGeometry {
    std::int32_t input_rows;
    std::int32_t heads;
};

GdnControlParentGeometry require_bf16_parent(const Weight& parent) {
    if (parent.n == 96 && parent.k == 5120) {
        require_bf16_weight(parent, 96, 5120, "ab_weight");
        return {.input_rows = 5120, .heads = 48};
    }
    if (parent.n == 64 && parent.k == 2048) {
        require_bf16_weight(parent, 64, 2048, "ab_weight");
        return {.input_rows = 2048, .heads = 32};
    }
    throw std::invalid_argument("gdn_gating_proj: unsupported ab_weight geometry");
}

void require_vector_tensor(const Tensor& t, DType dtype, std::int32_t n0, const char* op,
                           const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned_to(t.data, dtype == DType::FP32 ? 4 : 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void require_sequence_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t tokens,
                             const char* op, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned_to(t.data, dtype == DType::FP32 ? 4 : 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void require_execution(DeviceExecutionView execution, const char* op) {
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument(std::string(op) +
                                    ": execution multiprocessor count must be positive");
    }
}

} // namespace

std::size_t gdn_gating_proj_workspace_capacity_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    return detail::bf16_gdn_gating_capacity_workspace_bytes(heads, input_rows, min_tokens,
                                                            max_tokens);
}

std::size_t gdn_norm_gating_proj_workspace_capacity_bytes(std::int32_t heads,
                                                          std::int32_t input_rows,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    return detail::bf16_gdn_norm_gating_capacity_workspace_bytes(heads, input_rows, min_tokens,
                                                                 max_tokens);
}

void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, DeviceExecutionView execution) {
    constexpr const char* op  = "gdn_gating_proj";
    const std::int32_t tokens = x.ne[1];
    require_sequence_tensor(x, DType::BF16, 5120, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, 48, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, 48, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, 48, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, 48, tokens, op, "beta");
    require_bf16_weight(a_weight, 48, 5120, "a_weight");
    require_bf16_weight(b_weight, 48, 5120, "b_weight");
    require_execution(execution, op);

    detail::bf16_gdn_gating_dispatch(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, execution);
}

void gdn_gating_proj(const Tensor& x, const Weight& ab_weight, const Tensor& A_log,
                     const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g, Tensor& beta,
                     DeviceExecutionView execution) {
    constexpr const char* op                = "gdn_gating_proj";
    const std::int32_t tokens               = x.ne[1];
    const GdnControlParentGeometry geometry = require_bf16_parent(ab_weight);
    require_sequence_tensor(x, DType::BF16, geometry.input_rows, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, geometry.heads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, geometry.heads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, geometry.heads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, geometry.heads, tokens, op, "beta");
    require_execution(execution, op);

    const Weight a_weight = bf16_row_view(ab_weight, 0, geometry.heads);
    const Weight b_weight = bf16_row_view(ab_weight, geometry.heads, geometry.heads);
    detail::bf16_gdn_gating_dispatch(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, execution);
}

void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                          const Tensor& dt_bias, WorkspaceArena& ws, Tensor& h, Tensor& g,
                          Tensor& beta, DeviceExecutionView execution) {
    constexpr const char* op  = "gdn_norm_gating_proj";
    const std::int32_t tokens = x.ne[1];
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("gdn_norm_gating_proj: eps must be positive and finite");
    }
    require_sequence_tensor(x, DType::BF16, 5120, tokens, op, "x");
    require_vector_tensor(norm_weight, DType::BF16, 5120, op, "norm_weight");
    require_sequence_tensor(h, DType::BF16, 5120, tokens, op, "h");
    require_vector_tensor(A_log, DType::FP32, 48, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, 48, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, 48, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, 48, tokens, op, "beta");
    require_bf16_weight(a_weight, 48, 5120, "a_weight");
    require_bf16_weight(b_weight, 48, 5120, "b_weight");
    require_execution(execution, op);

    detail::bf16_gdn_norm_gating_dispatch(x, norm_weight, eps, h, a_weight, b_weight, A_log,
                                          dt_bias, ws, g, beta, execution);
}

void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& ab_weight, const Tensor& A_log, const Tensor& dt_bias,
                          WorkspaceArena& ws, Tensor& h, Tensor& g, Tensor& beta,
                          DeviceExecutionView execution) {
    constexpr const char* op  = "gdn_norm_gating_proj";
    const std::int32_t tokens = x.ne[1];
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("gdn_norm_gating_proj: eps must be positive and finite");
    }
    const GdnControlParentGeometry geometry = require_bf16_parent(ab_weight);
    require_sequence_tensor(x, DType::BF16, geometry.input_rows, tokens, op, "x");
    require_vector_tensor(norm_weight, DType::BF16, geometry.input_rows, op, "norm_weight");
    require_sequence_tensor(h, DType::BF16, geometry.input_rows, tokens, op, "h");
    require_vector_tensor(A_log, DType::FP32, geometry.heads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, geometry.heads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, geometry.heads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, geometry.heads, tokens, op, "beta");
    require_execution(execution, op);

    const Weight a_weight = bf16_row_view(ab_weight, 0, geometry.heads);
    const Weight b_weight = bf16_row_view(ab_weight, geometry.heads, geometry.heads);
    detail::bf16_gdn_norm_gating_dispatch(x, norm_weight, eps, h, a_weight, b_weight, A_log,
                                          dt_bias, ws, g, beta, execution);
}

namespace {

constexpr std::int32_t kShardHeads  = 24;
constexpr std::int32_t kShardHidden = 5120;

void validate_shard_rank(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                         const Tensor& A_log, const Tensor& dt_bias, const Tensor& g,
                         const Tensor& beta) {
    constexpr const char* op  = "gdn_gating_proj column-parallel";
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_sequence_tensor(x, DType::BF16, kShardHidden, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, kShardHeads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, kShardHeads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, kShardHeads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, kShardHeads, tokens, op, "beta");
    require_bf16_weight(a_weight, kShardHeads, kShardHidden, "a_weight shard");
    require_bf16_weight(b_weight, kShardHeads, kShardHidden, "b_weight shard");
}

void validate_shard_pair(const std::array<Tensor, 2>& x, const std::array<WorkspaceArena*, 2>& ws,
                         const ExecutionContext& ec) {
    detail::require_split_context(ec,
                                  "gdn_gating_proj column-parallel: requires two distinct devices");
    if (x[0].ne[1] != x[1].ne[1]) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel: ranks must agree on the token count");
    }
    if (detail::bf16_gdn_gating_shard_workspace_bytes(x[0].ne[1]) != 0 &&
        (ws[0] == nullptr || ws[1] == nullptr)) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel: T>=2 requires a workspace on every rank");
    }
}

void issue_shards(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& a_weight,
                  const std::array<Weight, 2>& b_weight, const std::array<Tensor, 2>& A_log,
                  const std::array<Tensor, 2>& dt_bias, const std::array<WorkspaceArena*, 2>& ws,
                  const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta,
                  const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, a_weight[slot].payload, g[slot].data,
            "gdn_gating_proj column-parallel: rank arguments must reside on its device");
        detail::require_rank_residency(
            ec, rank, dt_bias[slot].data, b_weight[slot].payload, beta[slot].data,
            "gdn_gating_proj column-parallel: rank arguments must reside on its device");
    }
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot         = static_cast<std::size_t>(rank);
        const std::size_t bytes = detail::bf16_gdn_gating_shard_workspace_bytes(x[slot].ne[1]);
        cudaStream_t stream     = ec.dev[slot]->stream;
        Tensor g_out            = g[slot];
        Tensor beta_out         = beta[slot];
        if (bytes == 0) {
            detail::bf16_gdn_gating_dispatch_shard(x[slot], a_weight[slot], b_weight[slot],
                                                   A_log[slot], dt_bias[slot], nullptr, 0, g_out,
                                                   beta_out, stream);
            return;
        }
        auto scope               = ws[slot]->scope();
        const DeviceSpan scratch = ws[slot]->alloc_bytes(bytes);
        detail::bf16_gdn_gating_dispatch_shard(x[slot], a_weight[slot], b_weight[slot], A_log[slot],
                                               dt_bias[slot], scratch.data, scratch.bytes, g_out,
                                               beta_out, stream);
    });
}

} // namespace

std::size_t gdn_gating_proj_column_parallel_workspace_capacity_bytes(std::int32_t heads,
                                                                     std::int32_t input_rows,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens) {
    if (heads != kShardHeads || input_rows != kShardHidden) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel workspace: unsupported shard profile");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel workspace: invalid token interval");
    }
    // The small-T scratch grows with T, so the interval's maximum is its last token count.
    return detail::bf16_gdn_gating_shard_workspace_bytes(max_tokens);
}

void gdn_gating_proj_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& a_weight,
    const std::array<Weight, 2>& b_weight, const std::array<Tensor, 2>& A_log,
    const std::array<Tensor, 2>& dt_bias, const std::array<WorkspaceArena*, 2>& ws,
    const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta, const ExecutionContext& ec) {
    validate_shard_pair(x, ws, ec);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        validate_shard_rank(x[slot], a_weight[slot], b_weight[slot], A_log[slot], dt_bias[slot],
                            g[slot], beta[slot]);
    }
    issue_shards(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, ec);
}

void gdn_gating_proj_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& ab_weight,
    const std::array<Tensor, 2>& A_log, const std::array<Tensor, 2>& dt_bias,
    const std::array<WorkspaceArena*, 2>& ws, const std::array<Tensor, 2>& g,
    const std::array<Tensor, 2>& beta, const ExecutionContext& ec) {
    validate_shard_pair(x, ws, ec);
    std::array<Weight, 2> a_weight{};
    std::array<Weight, 2> b_weight{};
    for (std::size_t slot = 0; slot < 2; ++slot) {
        require_bf16_weight(ab_weight[slot], 2 * kShardHeads, kShardHidden, "ab_weight shard");
        a_weight[slot] = bf16_row_view(ab_weight[slot], 0, kShardHeads);
        b_weight[slot] = bf16_row_view(ab_weight[slot], kShardHeads, kShardHeads);
        validate_shard_rank(x[slot], a_weight[slot], b_weight[slot], A_log[slot], dt_bias[slot],
                            g[slot], beta[slot]);
    }
    issue_shards(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, ec);
}

} // namespace ninfer::ops
