// Translation-unit fragment of src/ops/linear/linear.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) column- and row-parallel linear (ninfer/ops/tp2/linear.h), and the exported
// validation/dispatch pair of ops/linear/linear_dispatch.h that linear_add_row_parallel() uses for
// its residual-free rank.

#include "ninfer/ops/tp2/linear.h"
#include "ops/common/split_launch.h"
#include "ops/linear/linear_dispatch.h"

#include <array>
#include <cstddef>

namespace ninfer::ops {

namespace detail {

// linear.cpp keeps these two in its anonymous namespace; the qualified calls reach them there.
void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    ninfer::ops::validate_linear_semantics(x, w, out, policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    ninfer::ops::dispatch_linear(x, w, out, policy, workspace, stream);
}

} // namespace detail

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
    detail::require_allreduce_sum_arguments(out, staging, ec, events);
    // Each partial lands in out[r] on rank r's stream; allreduce_sum records its inputs_ready event
    // on that same stream, which orders the peer's read after the partial.
    issue_ranks(x, w, destination, policy, workspace, ec);
    allreduce_sum(out, staging, ec, events);
}

void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         const ExecutionContext& ec, const PeerEvents& events) {
    linear_row_parallel(x, w, out, staging, LinearPolicy::A16Only, {nullptr, nullptr}, ec, events);
}

} // namespace ninfer::ops
