#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include "ops/common/split_launch.h"
#include "ops/linear/linear_dispatch.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q8/q8_dispatch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

} // namespace

namespace detail {

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5_G64_FP16:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6_G64_FP16:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q8_G32_FP16:
        detail::q8_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace detail

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::Q4_G64_FP16:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5_G64_FP16:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6_G64_FP16:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q8_G32_FP16:
        (void)detail::select_q8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    detail::validate_linear_semantics(x, w, out, policy);
    detail::dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    detail::validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    detail::dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

namespace {

// Validates both ranks before either is issued, so a rejected pair enqueues nothing, and returns
// the mutable output views the dispatch takes. Only the pair can check the cross-rank agreement;
// the split axis itself may be uneven.
std::array<Tensor, 2> validate_split(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                                     const std::array<Tensor, 2>& out, LinearPolicy policy,
                                     const std::array<WorkspaceArena*, 2>& workspace,
                                     const ExecutionContext& ec, detail::SplitAxis axis) {
    const char* op =
        axis == detail::SplitAxis::Output ? "linear column-parallel" : "linear row-parallel";
    detail::require_split_pair(ec, x, w, axis, op);
    std::array<Tensor, 2> destination{out[0], out[1]};
    std::array<std::size_t, 2> required{};
    for (std::size_t rank = 0; rank < 2; ++rank) {
        detail::validate_linear_semantics(x[rank], w[rank], destination[rank], policy);
        detail::require_rank_residency(ec, static_cast<int>(rank), x[rank].data, w[rank].payload,
                                       out[rank].data,
                                       "linear split: rank arguments must reside on its device");
        required[rank] = linear_workspace_capacity_bytes(w[rank].qtype, w[rank].n, w[rank].k,
                                                         policy, x[rank].ne[1], x[rank].ne[1]);
    }
    detail::require_split_workspace(workspace, required, op);
    return destination;
}

void issue_ranks(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                 std::array<Tensor, 2>& out, LinearPolicy policy,
                 const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::dispatch_linear(x[slot], w[slot], out[slot], policy, workspace[slot],
                                ec.dev[slot]->stream);
    });
}

} // namespace

void linear_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& out, LinearPolicy policy,
                            const std::array<WorkspaceArena*, 2>& workspace,
                            const ExecutionContext& ec) {
    std::array<Tensor, 2> destination =
        validate_split(x, w, out, policy, workspace, ec, detail::SplitAxis::Output);
    issue_ranks(x, w, destination, policy, workspace, ec);
}

void linear_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& out, const ExecutionContext& ec) {
    linear_column_parallel(x, w, out, LinearPolicy::A16Only, {nullptr, nullptr}, ec);
}

void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         LinearPolicy policy, const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& ec, const PeerEvents& events) {
    std::array<Tensor, 2> destination =
        validate_split(x, w, out, policy, workspace, ec, detail::SplitAxis::Input);
    if (!events.live()) { throw std::invalid_argument("linear row-parallel: events must be live"); }
    // Each partial lands in out[r] on rank r's stream; allreduce_sum records its inputs_ready event
    // on that same stream, which orders the peer's read after the partial, and checks staging.
    issue_ranks(x, w, destination, policy, workspace, ec);
    allreduce_sum(out, staging, ec, events);
}

void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         const ExecutionContext& ec, const PeerEvents& events) {
    linear_row_parallel(x, w, out, staging, LinearPolicy::A16Only, {nullptr, nullptr}, ec, events);
}

} // namespace ninfer::ops
