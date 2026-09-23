#pragma once

#include "core/weight.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Computes four independent linear projections for each token:
 *
 *   q[:,t]    = linear(x[:,t], query_key_weight[0:6144,:])
 *   k[:,t]    = linear(x[:,t], query_key_weight[6144:7168,:])
 *   gate[:,t] = linear(x[:,t], gate_value_weight[0:6144,:])
 *   v[:,t]    = linear(x[:,t], gate_value_weight[6144:7168,:]).
 *
 * All tensors are contiguous BF16. Shapes are x [5120,T], q/gate [6144,T], and k/v [1024,T].
 * T may be any positive value.
 * The two parent weights are RowSplit [7168,5120] with FP16 scales and group size 64:
 * query_key is Q4_G64_FP16 and gate_value is Q5_G64_FP16. The oracle exact-decodes each row and
 * evaluates every projection naively in FP64 from the represented inputs. The BF16 outputs are
 * promoted and compared directly with those ideal values; final output storage rounding belongs
 * to AttnInputProj's named A16 criterion, not the oracle. Production routes choose their private
 * accumulator and staging precision. Inputs and the four outputs must be mutually non-overlapping.
 * Current registered routes require no transient allocation. The Op has no persistent state side
 * effect.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream);

/**
 * Computes the single-parent Q/K/output-gate/V projection.
 *
 * The parent stores rows in physical order query, key, output gate, value while the public output
 * argument order is q, gate, k, v. Every route writes the four independently contiguous final
 * allocations directly; no packed parent output is materialized. The NVFP4 A4 and FP8 A8
 * profiles may use caller-owned transient storage for their private quantized activation.
 *
 * Registered parent forms are:
 *
 * - Q8_G32_FP16 RowSplit `[9216,2048]`, with row counts `[4096,512,4096,512]`. `x` is
 *   BF16 `[2048,T]`, q/gate are BF16 `[4096,T]`, and k/v are BF16 `[512,T]`.
 * - BF16 Contiguous `[14336,5120]`, with row counts `[6144,1024,6144,1024]`. `x` is
 *   BF16 `[5120,T]`, q/gate are BF16 `[6144,T]`, and k/v are BF16 `[1024,T]`.
 * - NVFP4 BlockScaleK16M128x4 `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16.
 * - FP8_E4M3FN_ROW_BF16 RowScale `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16.
 *
 * `T` is the positive token extent of the Op contract. All three policies permit the BF16
 * and Q8_G32_FP16 A16 implementations. NVFP4 uses A16 under A16Only/AllowA8; AllowA4 permits the
 * resolver to select either a qualified A16 route or activation quantization to NVFP4 at every
 * positive T. FP8 accepts all policies at every positive T. AllowA8/AllowA4 permit the resolver to
 * choose a qualified A16 route or private activation quantization followed by A8 Tensor Core
 * computation. A16Only preserves the represented BF16 activation at every positive T; tile and
 * route cutoffs are private implementation choices, independent of speculative block width.
 *
 * The oracle evaluates every projection independently with naive FP64 accumulation from the
 * logical values represented by the persistent weight and BF16 activation. The final four BF16
 * stores belong to the Op's criterion for the selected activation-compute path.
 *
 * `workspace` is caller-owned call-scoped transient storage sized by
 * attn_input_proj_workspace_capacity_bytes(). It must not overlap the input, parent weight, or any
 * output. The Op does not allocate device memory internally.
 */
[[nodiscard]] std::size_t
attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                         std::int32_t input_rows, LinearPolicy policy,
                                         std::int32_t min_tokens, std::int32_t max_tokens);

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Applies the A16-only single-parent Q/K/output-gate/V projection without transient workspace.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

/**
 * Three-output Q8 specialization. The Q8_G32_FP16 RowSplit parent stores rows in order
 * [query 4096, key 1024, value 1024]. Registered parent forms are [6144,2048] with BF16
 * x [2048,T] for the Qwen3.6 companion and [6144,5120] with BF16 x [5120,T] for DFlash2.
 * q is contiguous BF16 [4096,T], and k/v are contiguous BF16 [1024,T]. Every route writes the
 * three independent final allocations directly; no parent output or transient workspace is
 * materialized. T may be any positive value. Q and K remain raw projection outputs: this Op does
 * not normalize or rotate either tensor.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream);

// Tensor-parallel form over two devices.
//
// The single-parent projection splits by heads. Rank r's weight is a standalone shard whose four
// sections are rank r's halves of the parent's query, key, gate and value sections, stacked in the
// parent's query, key, gate, value row order, so each rank computes its own heads' outputs with no
// communication. For the registered FP8 parent `[14336,5120]` the shard is `[7168,5120]` with row
// counts `[3072,512,3072,512]`: 12 of the 24 query/gate heads and 2 of the 4 key/value heads. The
// shard is never a view into the parent's payload.
//
// Every requirement of the single-parent attn_input_proj() applies per rank at the shard shapes.
// `x[r]`, `w[r]`, the four outputs and `workspace[r]` must be resident on `ec.dev[r]`, and rank r's
// work is enqueued on `ec.dev[r]->stream`. The caller obligation of ninfer/ops/allreduce.h applies:
// inputs staged on a device's legacy default stream must be retired before the call. The form does
// not synchronize, and the caller's current CUDA device is preserved.

/**
 * Returns the per-rank transient capacity attn_input_proj_column_parallel() requires for every T
 * in `[min_tokens,max_tokens]`. Only FP8_E4M3FN_ROW_BF16 shards are registered; the shard keeps the
 * parent's input rows, so this equals the parent's capacity.
 */
[[nodiscard]] std::size_t attn_input_proj_column_parallel_workspace_capacity_bytes(
    QType shard_qtype, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * @brief Column-parallel (head-split) single-parent projection across two devices.
 *
 * @details Rank r computes `q[r]`, `gate[r]`, `k[r]` and `v[r]` from `x[r]` and its shard
 * `query_key_gate_value_weight[r]`. Concatenating the ranks' outputs along `ne[0]` gives the
 * single-device outputs of the parent, and each rank's numerical contract is that of
 * attn_input_proj() under the same policy. Both ranks must agree on the weight format, on `K` and
 * on the token count `T`. Only FP8_E4M3FN_ROW_BF16 `[7168,5120]` shards are registered.
 *
 * @param[in] x Per-rank BF16 activation `[5120,T]`, identical on both ranks.
 * @param[in] query_key_gate_value_weight Per-rank FP8 shard `[7168,5120]`.
 * @param[out] q,gate Per-rank BF16 `[3072,T]`.
 * @param[out] k,v Per-rank BF16 `[512,T]`.
 * @param[in] policy Permitted private activation-compute profiles, applied to both ranks.
 * @param[in,out] workspace Per-rank caller-owned transient arena, sized by
 * attn_input_proj_column_parallel_workspace_capacity_bytes().
 * @param[in] ec Execution context holding two distinct devices.
 */
void attn_input_proj_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_gate_value_weight,
    const std::array<Tensor, 2>& q, const std::array<Tensor, 2>& gate,
    const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);

/// A16-only column-parallel form; it requires no transient workspace.
/// Model execution passes a policy; this form is the A16 entry the op qualification suites use.
void attn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& query_key_gate_value_weight,
                                     const std::array<Tensor, 2>& q,
                                     const std::array<Tensor, 2>& gate,
                                     const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v,
                                     const ExecutionContext& ec);

} // namespace ninfer::ops
