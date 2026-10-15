#pragma once

// Two-device (tp 2) forms of linear_add. Our file: ninfer/ops/linear_add.h is upstream's and carries no
// hook.
//
// Shapes registered for the two-device forms: NVFP4 and FP8 register the input-column halves
// [5120,3072] and [5120,8704] of [5120,6144] and [5120,17408], each keeping the activation-
// quantization crossover of the problem it halves; linear() over the same half switches at the
// same T, so both ranks of a row-parallel projection take the same route.

#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear_add.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the per-rank transient capacity linear_add_row_parallel() requires for every T in the
 * inclusive [min_tokens,max_tokens] interval, at the shard shape `[output_rows,input_rows]`. One
 * arena of this size serves either rank. Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t
linear_add_row_parallel_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                 std::int32_t input_rows, LinearPolicy policy,
                                                 std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * @brief Row-parallel (input-split) linear_add across two devices, summed across ranks.
 *
 * @details Rank r holds the activation block `x[r]` `[K_r,T]` and the matching weight-column
 * shard `w[r]` `[N,K_r]`, a standalone tensor of a registered linear_add problem. `residual[r]`
 * holds the same `[N,T]` residual on both ranks on entry. On completion both ranks hold
 *
 * @f[
 *   \mathrm{ideal}_{n,t} = \mathrm{residual}_{n,t} + \sum_{r} \sum_{k \in \mathrm{block}(r)}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * The residual enters the sum once: rank 0 runs linear_add() on its copy, rank 1 overwrites its
 * copy with the residual-free partial of linear() at the same shard shape, and one allreduce_sum()
 * combines the two. The result therefore carries the storage roundings of both partials and of the
 * sum, which the single-device evaluation does not; an A8 route also quantizes each rank's
 * activation block with that block's own per-token scale.
 *
 * Both ranks must agree on the weight format, on `N`, and on `T`. Every requirement of
 * linear_add() applies per rank. `x[r]`, `w[r]`, `residual[r]`, `staging[r]` and `workspace[r]`
 * must be resident on `ec.dev[r]`, and rank r's work is enqueued on `ec.dev[r]->stream`.
 * `staging[r]` is BF16 scratch of `residual[r]`'s shape that must not overlap it; its contents
 * after the call are unspecified. The caller obligation of ninfer/ops/allreduce.h applies, and
 * consecutive calls sharing buffers, staging and events need no host synchronization between them.
 * The call does not synchronize and preserves the caller's current CUDA device.
 *
 * @param[in] x Per-rank BF16 activation block `[K_r,T]`.
 * @param[in] w Per-rank weight-column shard `[N,K_r]`.
 * @param[in,out] residual Per-rank BF16 `[N,T]`, identical on entry; both ranks hold the identical
 * updated residual on completion.
 * @param[in,out] staging Per-rank BF16 scratch matching `residual`.
 * @param[in] policy Permitted private activation-compute profiles, applied to both ranks.
 * @param[in,out] workspace Per-rank caller-owned transient arena, sized by
 * linear_add_row_parallel_workspace_capacity_bytes(). It may be null when that capacity is zero.
 * @param[in] ec Execution context holding two distinct devices.
 * @param[in] events Live cross-device ordering events, as for allreduce_sum().
 */
void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, LinearPolicy policy,
                             const std::array<WorkspaceArena*, 2>& workspace,
                             const ExecutionContext& ec, const PeerEvents& events);

/// A16-only row-parallel form; it requires no transient workspace.
/// Model execution passes a policy; this form is the A16 entry the op qualification suites use.
void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, const ExecutionContext& ec,
                             const PeerEvents& events);

} // namespace ninfer::ops
