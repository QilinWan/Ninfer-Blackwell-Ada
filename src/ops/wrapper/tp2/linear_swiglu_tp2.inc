// Translation-unit fragment of src/ops/wrapper/linear_swiglu.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) column-parallel form of linear_swiglu, declared in ninfer/ops/tp2/linear_swiglu.h.

#include "ninfer/ops/tp2/linear_swiglu.h"
#include "ops/common/split_launch.h"

#include <array>
#include <cstddef>

namespace ninfer::ops {

void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x,
                                   const std::array<Weight, 2>& gate_up_weight,
                                   const std::array<Tensor, 2>& out, LinearPolicy policy,
                                   const std::array<WorkspaceArena*, 2>& workspace,
                                   const ExecutionContext& ec) {
    constexpr const char* kOp = "linear_swiglu column-parallel";
    detail::require_split_pair(ec, x, gate_up_weight, detail::SplitAxis::Output, kOp);
    // Both ranks are validated before either issues work, so a rejected pair enqueues nothing.
    std::array<Tensor, 2> destination{out[0], out[1]};
    std::array<std::size_t, 2> required{};
    for (std::size_t rank = 0; rank < 2; ++rank) {
        validate_linear_swiglu(x[rank], gate_up_weight[rank], destination[rank], policy);
        detail::require_rank_residency(
            ec, static_cast<int>(rank), x[rank].data, gate_up_weight[rank].payload, out[rank].data,
            "linear_swiglu column-parallel: rank arguments must reside on its device");
        required[rank] = linear_swiglu_workspace_capacity_bytes(
            gate_up_weight[rank].qtype, gate_up_weight[rank].n, gate_up_weight[rank].k, policy,
            x[rank].ne[1], x[rank].ne[1]);
    }
    detail::require_split_workspace(workspace, required, kOp);
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
