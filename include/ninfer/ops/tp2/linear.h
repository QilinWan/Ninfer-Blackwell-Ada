#pragma once

// Two-device (tp 2) forms of linear. Our file: ninfer/ops/linear.h is upstream's and carries no
// hook.
//
// Shapes registered for the two-device forms (the upstream header documents the single-device
// problems; each half resolves to the routes of the problem it halves): FP8 registers the halves
// [7168,5120], [8192,5120], [17408,5120], [124160,5120], [5120,3072] and [5120,8704]; NVFP4
// [17408,5120], [5120,3072] and [5120,8704]; BF16 [7168,5120] and [5120,3072]; Q8 the MTP halves
// [5120,5120], [7168,5120], [17408,5120], [5120,3072] and [5120,8704]; Q4 [65536,5120] (the half of
// the [131072,5120] proposal head); each at every positive T. FP8 [5120,3072] takes A8 from T=22,
// as linear_add() over the same half does (NVFP4 [5120,3072] takes A4 from T=17 like both its
// parent and linear_add()).

#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Tensor-parallel forms over two devices.
//
// Both forms compose the single-device linear() above; neither is a separate kernel. A shard is a
// standalone `[N,K]` weight of the same registered format with one axis narrowed, never a view
// into its parent's payload, so the existing kernel run against a shard reads only that device's
// bytes and resolves the shard's own registered problem. The shard shapes must therefore be
// registered problems of the weight's format; an unregistered shard shape is rejected exactly as
// an unregistered whole shape is.
//
// Every requirement of linear() applies per rank. `x[r]`, `w[r]`, `out[r]`, `staging[r]` and
// `workspace[r]` must be resident on `ec.dev[r]`, and rank r's work is enqueued on
// `ec.dev[r]->stream`. The caller obligation of ninfer/ops/allreduce.h applies: inputs staged on a
// device's legacy default stream must be retired before the call. Neither form synchronizes, and
// the caller's current CUDA device is preserved.

/**
 * @brief Column-parallel (output-split) projection across two devices.
 *
 * @details Rank r computes `out[r] = w[r] x[r]`, where `w[r]` is rank r's contiguous block of the
 * logical weight's output rows and `x[r]` holds the same activation on both ranks. The two output
 * blocks concatenate along `ne[0]` into the single-device result. Nothing is communicated, so each
 * rank's numerical contract is that of linear() at the shard shape.
 *
 * Both ranks must agree on the weight format, on `K`, and on the token count `T`. The per-rank
 * output row counts need not be equal.
 *
 * @param[in] x Per-rank BF16 activation `[K,T]`, identical on both ranks.
 * @param[in] w Per-rank weight-row shard `[N_r,K]`.
 * @param[out] out Per-rank BF16 output block `[N_r,T]`.
 * @param[in] policy Permitted private activation-compute profiles, applied to both ranks.
 * @param[in,out] workspace Per-rank caller-owned transient arena, sized by
 * linear_workspace_capacity_bytes() at the shard shape.
 * @param[in] ec Execution context holding two distinct devices.
 */
void linear_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& out, LinearPolicy policy,
                            const std::array<WorkspaceArena*, 2>& workspace,
                            const ExecutionContext& ec);

/// A16-only column-parallel form; it requires no transient workspace.
/// Model execution passes a policy; this form is the A16 entry the op qualification suites use.
void linear_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& out, const ExecutionContext& ec);

/**
 * @brief Row-parallel (input-split) projection across two devices, summed across ranks.
 *
 * @details Rank r computes the full-width partial `w[r] x[r]`, where `w[r]` is rank r's block of
 * the logical weight's input columns and `x[r]` the matching block of the activation rows. One
 * allreduce_sum() then leaves the complete `[N,T]` result on both ranks:
 *
 * @f[
 *   \mathrm{ideal}_{n,t} = \sum_{r} \sum_{k \in \mathrm{block}(r)}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * Each rank rounds its partial to BF16 before the sum, so the result carries two storage roundings
 * of partial magnitude that the single-device evaluation does not. An A8 route also quantizes each
 * rank's activation block with that block's own per-token scale.
 *
 * Both ranks must agree on the weight format, on `N`, and on the token count `T`. The per-rank
 * input extents need not be equal. `staging[r]` is BF16 scratch of `out[r]`'s shape that must not
 * overlap it; its contents after the call are unspecified. `events` must be live. Consecutive
 * calls sharing buffers, staging and events need no host synchronization between them.
 *
 * @param[in] x Per-rank BF16 activation block `[K_r,T]`.
 * @param[in] w Per-rank weight-column shard `[N,K_r]`.
 * @param[in,out] out Per-rank BF16 `[N,T]`; both ranks hold the identical sum on completion.
 * @param[in,out] staging Per-rank BF16 scratch matching `out`.
 * @param[in] policy Permitted private activation-compute profiles, applied to both ranks.
 * @param[in,out] workspace Per-rank caller-owned transient arena, sized at the shard shape.
 * @param[in] ec Execution context holding two distinct devices.
 * @param[in] events Live cross-device ordering events, as for allreduce_sum().
 */
void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         LinearPolicy policy, const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& ec, const PeerEvents& events);

/// A16-only row-parallel form; it requires no transient workspace.
/// Model execution passes a policy; this form is the A16 entry the op qualification suites use.
void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         const ExecutionContext& ec, const PeerEvents& events);

} // namespace ninfer::ops
