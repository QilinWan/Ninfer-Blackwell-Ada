// Translation-unit fragment of src/ops/wrapper/linear_add.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) row-parallel form of linear_add, declared in ninfer/ops/tp2/linear_add.h.

#include "ninfer/ops/tp2/linear_add.h"
#include "ops/common/split_launch.h"
#include "ops/linear/linear_dispatch.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace ninfer::ops {

std::size_t linear_add_row_parallel_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                             std::int32_t input_rows,
                                                             LinearPolicy policy,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens) {
    // Rank 0 runs linear_add() and rank 1 linear() at the shard shape; one arena size serves both.
    return std::max(linear_add_workspace_capacity_bytes(qtype, output_rows, input_rows, policy,
                                                        min_tokens, max_tokens),
                    linear_workspace_capacity_bytes(qtype, output_rows, input_rows, policy,
                                                    min_tokens, max_tokens));
}

void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, LinearPolicy policy,
                             const std::array<WorkspaceArena*, 2>& workspace,
                             const ExecutionContext& ec, const PeerEvents& events) {
    constexpr const char* kOp = "linear_add row-parallel";
    detail::require_split_pair(ec, x, w, detail::SplitAxis::Input, kOp);
    if (!events.live()) {
        throw std::invalid_argument("linear_add row-parallel: events must be live");
    }
    // Both ranks are validated before either issues work, so a rejected pair enqueues nothing.
    std::array<Tensor, 2> destination{residual[0], residual[1]};
    for (std::size_t rank = 0; rank < 2; ++rank) {
        validate_linear_add(x[rank], w[rank], destination[rank], policy);
        detail::require_rank_residency(
            ec, static_cast<int>(rank), x[rank].data, w[rank].payload, residual[rank].data,
            "linear_add row-parallel: rank arguments must reside on its device");
    }
    // Rank 0 runs linear_add() and rank 1 linear(), each at its own route.
    const std::int32_t tokens = x[0].ne[1];
    detail::require_split_workspace(
        workspace,
        {linear_add_workspace_capacity_bytes(w[0].qtype, w[0].n, w[0].k, policy, tokens, tokens),
         linear_workspace_capacity_bytes(w[1].qtype, w[1].n, w[1].k, policy, tokens, tokens)},
        kOp);
    detail::require_allreduce_sum_arguments(residual, staging, ec, events);
    // The residual must enter the sum once: rank 0 adds its partial into its copy, and rank 1
    // overwrites its copy with the residual-free partial. allreduce_sum() then leaves
    // `residual + partial_0 + partial_1` on both ranks. It records its inputs-ready event on each
    // rank's stream after the partial, which orders the peer's read.
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot           = static_cast<std::size_t>(rank);
        const cudaStream_t stream = ec.dev[slot]->stream;
        if (rank == 0) {
            dispatch_linear_add(x[slot], w[slot], destination[slot], policy, workspace[slot],
                                stream);
        } else {
            detail::dispatch_linear(x[slot], w[slot], destination[slot], policy, workspace[slot],
                                    stream);
        }
    });
    allreduce_sum(residual, staging, ec, events);
}

void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, const ExecutionContext& ec,
                             const PeerEvents& events) {
    linear_add_row_parallel(x, w, residual, staging, LinearPolicy::A16Only, {nullptr, nullptr}, ec,
                            events);
}

} // namespace ninfer::ops
