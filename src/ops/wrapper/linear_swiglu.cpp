#include "core/weight.h"
#include "ninfer/ops/linear_swiglu.h"

#include "ops/common/split_launch.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

// The NVFP4 and FP8 gate/up problems: the single-device [34816,5120] and its two-device
// output-row half [17408,5120].
bool fused_gate_up_problem(std::int32_t gate_up_rows, std::int32_t input_rows) {
    return (gate_up_rows == 34816 || gate_up_rows == 17408) && input_rows == 5120;
}

} // namespace

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (qtype == QType::Q8_G32_FP16) {
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q4_G64_FP16) {
        return detail::q4_linear_swiglu_capacity_workspace_bytes(
            gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4 && fused_gate_up_problem(gate_up_rows, input_rows)) {
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(gate_up_rows, policy,
                                                                    min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16 && fused_gate_up_problem(gate_up_rows, input_rows)) {
        return detail::fp8_linear_swiglu_workspace_capacity_bytes(policy, gate_up_rows, min_tokens,
                                                                  max_tokens);
    }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

namespace {

bool matches_problem(const Tensor& x, const Weight& w, const Tensor& out, std::int32_t gate_up_rows,
                     std::int32_t input_rows) {
    return x.ne[0] == input_rows && out.ne[0] == gate_up_rows / 2 && w.n == gate_up_rows &&
           w.k == input_rows && w.padded_shape[0] == gate_up_rows &&
           w.padded_shape[1] == input_rows;
}

// Every check linear_swiglu() makes, so that a split form can reject a rank pair before either
// rank issues work.
void validate_linear_swiglu(const Tensor& x, const Weight& gate_up_weight, const Tensor& out,
                            LinearPolicy policy) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t   = x.ne[1];
    const bool large_shape = matches_problem(x, gate_up_weight, out, 34816, 5120);
    const bool shard_shape = matches_problem(x, gate_up_weight, out, 17408, 5120);
    const bool q8_shape    = matches_problem(x, gate_up_weight, out, 12288, 2048);
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !shard_shape && !q8_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight = large_shape && gate_up_weight.qtype == QType::Q4_G64_FP16 &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                           common_row_split;
    const bool q8_weight =
        (q8_shape || large_shape) && gate_up_weight.qtype == QType::Q8_G32_FP16 &&
        gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
        gate_up_weight.qhigh == nullptr && gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool fused_shape  = large_shape || shard_shape;
    const bool nvfp4_weight = fused_shape && gate_up_weight.qtype == QType::NVFP4;
    const bool fp8_weight   = fused_shape && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16;
    if (!q4_weight && !q8_weight && !nvfp4_weight && !fp8_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (fp8_weight) {
        (void)detail::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        return;
    }

    if (nvfp4_weight) {
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
        return;
    }

    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, q8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }
}

// Issues a validated call. `ws` may be null when the resolved route needs no workspace.
void dispatch_linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                            LinearPolicy policy, WorkspaceArena* ws, cudaStream_t stream) {
    switch (gate_up_weight.qtype) {
    case QType::FP8_E4M3FN_ROW_BF16:
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    case QType::Q8_G32_FP16:
        detail::q8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
        return;
    case QType::Q4_G64_FP16:
        if (ws == nullptr) {
            throw std::invalid_argument("linear_swiglu: Q4 requires caller workspace");
        }
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, *ws, stream);
        return;
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::BF16:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear_swiglu: unsupported weight");
}

} // namespace

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_linear_swiglu(x, gate_up_weight, out, policy);
    dispatch_linear_swiglu(x, gate_up_weight, out, policy, &ws, stream);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x,
                                   const std::array<Weight, 2>& gate_up_weight,
                                   const std::array<Tensor, 2>& out, LinearPolicy policy,
                                   const std::array<WorkspaceArena*, 2>& workspace,
                                   const ExecutionContext& ec) {
    detail::require_split_context(ec,
                                  "linear_swiglu column-parallel: requires two distinct devices");
    if (x[0].ne[1] != x[1].ne[1]) {
        throw std::invalid_argument("linear_swiglu column-parallel: ranks must agree on T");
    }
    if (gate_up_weight[0].qtype != gate_up_weight[1].qtype ||
        gate_up_weight[0].layout != gate_up_weight[1].layout) {
        throw std::invalid_argument(
            "linear_swiglu column-parallel: ranks must agree on the weight format");
    }
    if (gate_up_weight[0].k != gate_up_weight[1].k) {
        throw std::invalid_argument("linear_swiglu column-parallel: ranks must agree on K");
    }
    // Both ranks are validated before either issues work, so a rejected pair enqueues nothing.
    std::array<Tensor, 2> destination{out[0], out[1]};
    for (std::size_t rank = 0; rank < 2; ++rank) {
        validate_linear_swiglu(x[rank], gate_up_weight[rank], destination[rank], policy);
        detail::require_rank_residency(
            ec, static_cast<int>(rank), x[rank].data, gate_up_weight[rank].payload, out[rank].data,
            "linear_swiglu column-parallel: rank arguments must reside on its device");
    }
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        dispatch_linear_swiglu(x[slot], gate_up_weight[slot], destination[slot], policy,
                               workspace[slot], ec.dev[slot]->stream);
    });
}

void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x,
                                   const std::array<Weight, 2>& gate_up_weight,
                                   const std::array<Tensor, 2>& out, const ExecutionContext& ec) {
    linear_swiglu_column_parallel(x, gate_up_weight, out, LinearPolicy::A16Only, {nullptr, nullptr},
                                  ec);
}

} // namespace ninfer::ops
