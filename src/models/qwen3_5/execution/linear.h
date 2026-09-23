#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear(input, p.weight, output, p.policy, workspace, stream);
}

inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_add(input, p.weight, residual, p.policy, workspace, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_swiglu(input, p.weight, output, p.policy, workspace, stream);
}

// Tensor-parallel projections over two ranks. Rank r's operand is its own shard of one logical
// projection; the split Ops apply one activation policy to both ranks, so the two shards must
// carry the same one (they are two slices of one stored parent and its Use).

inline ops::LinearPolicy split_policy(const std::array<const LinearParameters*, 2>& p,
                                      const char* label) {
    if (p[0] == nullptr || p[1] == nullptr) {
        throw std::invalid_argument(std::string(label) + ": a rank has no projection shard");
    }
    if (p[0]->policy != p[1]->policy) {
        throw std::invalid_argument(std::string(label) +
                                    ": ranks disagree on the activation policy");
    }
    return p[0]->policy;
}

inline std::array<Weight, 2> split_weights(const std::array<const LinearParameters*, 2>& p) {
    return {p[0]->weight, p[1]->weight};
}

// Output-split projection: out[r] = w[r] x[r], x replicated.
inline void project_column_parallel(const std::array<Tensor, 2>& input,
                                    const std::array<const LinearParameters*, 2>& p,
                                    const std::array<Tensor, 2>& output,
                                    const std::array<WorkspaceArena*, 2>& workspace,
                                    const ExecutionContext& execution) {
    const auto policy = split_policy(p, "column-parallel projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    ops::linear_column_parallel(input, split_weights(p), output, policy, workspace, execution);
}

// Input-split projection added to the replicated residual: one all-reduce, after which both ranks
// hold residual + w x. `staging[r]` matches `residual[r]` and must not overlap it.
inline void project_add_row_parallel(const std::array<Tensor, 2>& input,
                                     const std::array<const LinearParameters*, 2>& p,
                                     const std::array<Tensor, 2>& residual,
                                     const std::array<Tensor, 2>& staging,
                                     const std::array<WorkspaceArena*, 2>& workspace,
                                     const ExecutionContext& execution,
                                     const ops::PeerEvents& events) {
    const auto policy = split_policy(p, "row-parallel projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    ops::linear_add_row_parallel(input, split_weights(p), residual, staging, policy, workspace,
                                 execution, events);
}

// Input-split projection: one all-reduce, after which both ranks hold the complete w x in
// `output[r]`. `staging[r]` matches `output[r]` and must not overlap it.
inline void project_row_parallel(const std::array<Tensor, 2>& input,
                                 const std::array<const LinearParameters*, 2>& p,
                                 const std::array<Tensor, 2>& output,
                                 const std::array<Tensor, 2>& staging,
                                 const std::array<WorkspaceArena*, 2>& workspace,
                                 const ExecutionContext& execution, const ops::PeerEvents& events) {
    const auto policy = split_policy(p, "row-parallel projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    ops::linear_row_parallel(input, split_weights(p), output, staging, policy, workspace, execution,
                             events);
}

// Output-split gate/up projection with SwiGLU: out[r] is rank r's block of the intermediate
// activation, which is directly its input block of the row-parallel down projection.
inline void project_swiglu_column_parallel(const std::array<Tensor, 2>& input,
                                           const std::array<const LinearParameters*, 2>& p,
                                           const std::array<Tensor, 2>& output,
                                           const std::array<WorkspaceArena*, 2>& workspace,
                                           const ExecutionContext& execution) {
    const auto policy = split_policy(p, "column-parallel SwiGLU projection");
    auto scope0       = workspace[0]->scope();
    auto scope1       = workspace[1]->scope();
    ops::linear_swiglu_column_parallel(input, split_weights(p), output, policy, workspace,
                                       execution);
}

} // namespace ninfer::models::qwen3_5::execution
