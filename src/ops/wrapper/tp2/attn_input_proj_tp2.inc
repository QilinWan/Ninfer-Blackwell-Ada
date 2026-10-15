// Translation-unit fragment of src/ops/wrapper/attn_input_proj.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) forms of attn_input_proj, declared in ninfer/ops/tp2/attn_input_proj.h.

#include "ninfer/ops/tp2/attn_input_proj.h"
#include "ops/attn_input_proj/tp2/fp8_attn_input_shard.h"
#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard.h"
#include "ops/common/split_launch.h"

#include <array>

namespace ninfer::ops {
namespace {

// One rank of the column-parallel form: an FP8 or NVFP4 [7168,5120] head-local shard of the fused
// parent.
void validate_column_parallel_rank(const Tensor& x, const Weight& weight, const Tensor& q,
                                   const Tensor& gate, const Tensor& k, const Tensor& v,
                                   LinearPolicy policy) {
    validate_policy(policy);
    static_assert(detail::Fp8N7168K5120::kOutputRows == detail::Nvfp4N7168K5120::kOutputRows &&
                  detail::Fp8N7168K5120::kInputRows == detail::Nvfp4N7168K5120::kInputRows);
    constexpr std::int32_t kHidden = detail::Fp8N7168K5120::kInputRows;
    constexpr std::int32_t kQRows  = 3072;
    constexpr std::int32_t kKvRows = 512;
    constexpr std::int32_t kRows   = detail::Fp8N7168K5120::kOutputRows;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    if (weight.qtype == QType::NVFP4) {
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj column-parallel");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 attn_input_proj column-parallel: unsupported shard shape");
        }
        return;
    }
    if (weight.qtype != QType::FP8_E4M3FN_ROW_BF16) {
        throw std::invalid_argument("attn_input_proj column-parallel: only FP8_E4M3FN_ROW_BF16 and "
                                    "NVFP4 shards are registered");
    }
    detail::validate_fp8_weight(weight, "fp8 attn_input_proj column-parallel");
    if (weight.n != kRows || weight.k != kHidden) {
        throw std::invalid_argument("fp8 attn_input_proj column-parallel: unsupported shard shape");
    }
}

} // namespace

std::size_t attn_input_proj_column_parallel_workspace_capacity_bytes(QType shard_qtype,
                                                                     LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    // The shard keeps the parent's input rows; the FP8 workspace also holds its split-K partials.
    switch (shard_qtype) {
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_attn_input_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                     max_tokens);
    case QType::NVFP4:
        return detail::nvfp4_attn_input_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                       max_tokens);
    case QType::BF16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument(
        "attn_input_proj column-parallel workspace: unsupported shard qtype");
}

void attn_input_proj_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_gate_value_weight,
    const std::array<Tensor, 2>& q, const std::array<Tensor, 2>& gate,
    const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    const std::array<Weight, 2>& w = query_key_gate_value_weight;
    constexpr const char* kOp      = "attn_input_proj column-parallel";
    detail::require_split_pair(ec, x, w, detail::SplitAxis::Output, kOp);
    // Validate both ranks before issuing either, so a rejected pair enqueues nothing.
    for (std::size_t rank = 0; rank < 2; ++rank) {
        validate_column_parallel_rank(x[rank], w[rank], q[rank], gate[rank], k[rank], v[rank],
                                      policy);
        constexpr const char* kResidency =
            "attn_input_proj split: rank arguments must reside on its device";
        detail::require_rank_residency(ec, static_cast<int>(rank), x[rank].data, w[rank].payload,
                                       q[rank].data, kResidency);
        detail::require_rank_residency(ec, static_cast<int>(rank), gate[rank].data, k[rank].data,
                                       v[rank].data, kResidency);
    }
    const std::size_t bytes = attn_input_proj_column_parallel_workspace_capacity_bytes(
        w[0].qtype, policy, x[0].ne[1], x[0].ne[1]);
    detail::require_split_workspace(workspace, {bytes, bytes}, kOp);
    std::array<Tensor, 2> q_out{q[0], q[1]};
    std::array<Tensor, 2> gate_out{gate[0], gate[1]};
    std::array<Tensor, 2> k_out{k[0], k[1]};
    std::array<Tensor, 2> v_out{v[0], v[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (w[slot].qtype == QType::NVFP4) {
            detail::nvfp4_attn_input_shard_dispatch(x[slot], w[slot], q_out[slot], gate_out[slot],
                                                    k_out[slot], v_out[slot], policy,
                                                    workspace[slot], ec.dev[slot]->stream);
        } else {
            detail::fp8_attn_input_shard_dispatch(x[slot], w[slot], q_out[slot], gate_out[slot],
                                                  k_out[slot], v_out[slot], policy,
                                                  workspace[slot], ec.dev[slot]->stream);
        }
    });
}

void attn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& query_key_gate_value_weight,
                                     const std::array<Tensor, 2>& q,
                                     const std::array<Tensor, 2>& gate,
                                     const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v,
                                     const ExecutionContext& ec) {
    attn_input_proj_column_parallel(x, query_key_gate_value_weight, q, gate, k, v,
                                    LinearPolicy::A16Only, {nullptr, nullptr}, ec);
}

} // namespace ninfer::ops
