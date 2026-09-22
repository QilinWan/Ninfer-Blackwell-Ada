#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/weight.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the A16-only transient capacity required by LinearAdd for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   ideal[:,t] = residual[:,t] + Linear(x,w)[:,t].
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Registered weights are Q4_G64_FP16 RowSplit
 *   [5120,6144], Q5_G64_FP16 RowSplit [5120,17408] or [5120,6144], Q8_G32_FP16 RowSplit
 *   [2048,4096], [2048,6144], [5120,6144] or [5120,17408], NVFP4
 *   BlockScaleK16M128x4 [5120,6144] or [5120,17408], row-scaled
 *   FP8_E4M3FN_ROW_BF16 [5120,6144] or [5120,17408], or BF16 Contiguous [5120,6144]. For
 *   two-device execution, NVFP4 also registers the input-column half [5120,8704] and FP8 the
 *   halves [5120,3072] and [5120,8704], each resolving to the routes of the problem it halves.
 *   T may be any positive value.
 *
 * Numeric:
 *   The oracle reads a registered BF16 weight directly or exact-decodes a registered packed
 *   weight, then evaluates `ideal` naively in FP64 from the represented inputs. The updated BF16
 *   residual is promoted and compared directly with that result; output storage rounding belongs
 *   to LinearAdd's selected A16, A8, or A4 criterion, not the oracle. Production routes may fuse
 *   or materialize the projection and may choose their natural accumulator, activation
 *   quantization, staging, and workspace precision; those private choices are not semantic
 *   rounding boundaries.
 *
 * Compute policy:
 *   All policies permit the A16 implementations of Q4, Q5, Q8 and BF16. NVFP4 uses A16 for
 *   A16Only/AllowA8 and may use A4 under AllowA4. FP8 may use A8 under AllowA8/AllowA4.
 *   Each registration owns its production plan. A permissive policy
 *   allows the private resolver to select either qualified
 *   arithmetic profile; it does not itself prescribe a kernel.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_capacity_bytes(), scoped to
 *   the call. A16 routes require no storage; quantized-activation routes use the reported capacity.
 *   There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream);

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
void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, const ExecutionContext& ec,
                             const PeerEvents& events);

} // namespace ninfer::ops
