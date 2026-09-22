#pragma once

// ninfer::ops - fused gate/up projection followed by SwiGLU.

#include "core/weight.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient capacity required by LinearSwiGLU for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Policy-bearing capacity query. Q4/Q8 use A16 under every policy. NVFP4 uses A16 under
 * A16Only/AllowA8 through T=16; AllowA4 accepts every positive T. Row-scaled FP8 accepts all
 * policies, with A8 permitted by AllowA8/AllowA4.
 * A permissive policy covers whichever qualified route the private resolver selects across the
 * requested interval.
 */
[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = Linear(x, gate_up_weight); M=gate_up_rows/2;
 *   ideal[i,t] = SiLU(gate_up[i,t]) * gate_up[M+i,t].
 *
 * Logical shapes / supported domain:
 *   T may be any positive value. The registered profiles are:
 *   - Q4_G64_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - Q8_G32_FP16 weight [12288,2048], x [2048,T], out [6144,T];
 *   - Q8_G32_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - NVFP4 BlockScaleK16M128x4 weight [34816,5120], x [5120,T], out [17408,T];
 *   - FP8_E4M3FN_ROW_BF16 RowScale weight [34816,5120], x [5120,T], out [17408,T];
 *   - for two-device execution, NVFP4 and FP8 also register the output-row half: weight
 *     [17408,5120], x [5120,T], out [8704,T], with the same routes as the whole problem.
 *   Inputs and output are contiguous BF16. Q4/Q8 scales are FP16, NVFP4 scales are E4M3FN, and
 *   row-scaled FP8 has one BF16 multiplier per gate/up parent row. Gate rows `[0,M)` precede
 *   their matching up rows `[M,2M)`, M = N/2.
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates `ideal` naively in FP64 from the
 *   represented inputs. The BF16 output is promoted and compared directly with that result; output
 *   storage rounding belongs to LinearSwiGLU's named activation-compute criterion, not the oracle.
 *   Production routes may fuse or materialize gate/up and may choose their natural accumulator,
 *   staging, and workspace precision; those private choices are not semantic rounding boundaries.
 *   AllowA8/AllowA4 permit FP8 activation quantization; route thresholds are implementation
 * choices.
 *
 * Effects:
 *   Writes the full output; x/weight and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_swiglu_workspace_capacity_bytes(),
 *   scoped to the call. Q8, NVFP4 A16, and row-scaled FP8 A16 require zero bytes; A4/A8 routes use
 *   caller-owned activation storage and may use private projection storage. There is no persistent
 *   state side effect.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream);

/**
 * A16-only convenience form. Q4/Q8 and row-scaled FP8 retain their complete positive-T domain.
 * NVFP4 is admitted only through T=16; larger NVFP4 extents require the policy-bearing AllowA4
 * form.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

/**
 * @brief Column-parallel (output-split) linear_swiglu across two devices.
 *
 * @details Rank r's shard `gate_up_weight[r]` holds its block of the gate rows followed by the
 * matching block of the up rows: for the [34816,5120] problem split evenly, gate rows
 * `[r*8704,(r+1)*8704)` then up rows `[17408+r*8704,17408+(r+1)*8704)`, a standalone
 * [17408,5120] tensor in the single-device gate/up layout. Rank r computes
 * `out[r] = linear_swiglu(x[r], gate_up_weight[r])` from the activation replicated on both ranks;
 * the two output blocks concatenate along `ne[0]` into the single-device result, and each block is
 * directly rank r's input block of the row-parallel down projection. Nothing is communicated, so
 * each rank's numerical contract is that of linear_swiglu() at the shard shape, which must be a
 * registered problem.
 *
 * Both ranks must agree on the weight format, on `K`, and on `T`. Every requirement of
 * linear_swiglu() applies per rank; `x[r]`, `gate_up_weight[r]`, `out[r]` and `workspace[r]` must
 * be resident on `ec.dev[r]`, and rank r's work is enqueued on `ec.dev[r]->stream`. The caller
 * obligation of ninfer/ops/allreduce.h applies: inputs staged on a device's legacy default stream
 * must be retired before the call. The call does not synchronize and preserves the caller's
 * current CUDA device.
 *
 * @param[in] x Per-rank BF16 activation `[K,T]`, identical on both ranks.
 * @param[in] gate_up_weight Per-rank gate/up shard `[N_r,K]`.
 * @param[out] out Per-rank BF16 output block `[N_r/2,T]`.
 * @param[in] policy Permitted private activation-compute profiles, applied to both ranks.
 * @param[in,out] workspace Per-rank caller-owned transient arena, sized by
 * linear_swiglu_workspace_capacity_bytes() at the shard shape. It may be null when that capacity
 * is zero.
 * @param[in] ec Execution context holding two distinct devices.
 */
void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x,
                                   const std::array<Weight, 2>& gate_up_weight,
                                   const std::array<Tensor, 2>& out, LinearPolicy policy,
                                   const std::array<WorkspaceArena*, 2>& workspace,
                                   const ExecutionContext& ec);

/// A16-only column-parallel form; it passes no workspace, which the NVFP4 and FP8 A16 routes do
/// not need.
void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x,
                                   const std::array<Weight, 2>& gate_up_weight,
                                   const std::array<Tensor, 2>& out, const ExecutionContext& ec);

} // namespace ninfer::ops
