// Translation-unit fragment of src/ops/wrapper/gdn_gating_proj.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) forms of gdn_gating_proj, declared in ninfer/ops/tp2/gdn_gating_proj.h.

#include "ninfer/ops/tp2/gdn_gating_proj.h"
#include "ops/common/split_launch.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include <array>
#include <cstddef>

namespace ninfer::ops {

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
    constexpr const char* kOp = "gdn_gating_proj column-parallel";
    detail::require_split_ranks(ec, x, kOp);
    const std::size_t bytes = detail::bf16_gdn_gating_shard_workspace_bytes(x[0].ne[1]);
    detail::require_split_workspace(ws, {bytes, bytes}, kOp);
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
