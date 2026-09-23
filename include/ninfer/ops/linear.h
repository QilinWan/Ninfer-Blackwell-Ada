#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit A16, A8 or A4 compute profiles.
};

[[nodiscard]] constexpr bool valid_linear_policy(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
           policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8 || policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a4(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA4;
}

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * @brief Applies a bias-free matrix projection independently to every input column.
 *
 * @details The ideal mathematical result is
 *
 * @f[
 *   \mathrm{ideal}_{n,t} =
 *   \sum_{k=0}^{K-1}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `out` stores a BF16 approximation of this ideal result under the named numerical criterion for
 * the selected private activation-compute path.
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 `[N,T]`. Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses RowSplit Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, or Q8_G32_FP16 weights
 * with FP16 scales, block-scaled NVFP4 weights, row-scaled FP8_E4M3FN_ROW_BF16 weights, plus
 * registered contiguous BF16 problems. Each format owns a finite registry of exact physical
 * weight problems and selects its kernel internally; a valid encoding and alignment do not imply
 * arbitrary N/K support. FP8 currently registers `[N,K]` in `{[14336,5120], [16384,5120],
 * [34816,5120], [248320,5120], [5120,6144], [5120,17408]}` at every positive T, and for two-device
 * execution also registers their halves `[7168,5120]`, `[8192,5120]`, `[17408,5120]`,
 * `[124160,5120]`, `[5120,3072]` and `[5120,8704]`, each at every positive T and resolving to the
 * routes of the problem it halves, except that FP8 `[5120,8704]` takes A8 from T=20, as
 * linear_add() over the same half does (the `[5120,3072]` half needs no exception: its A8 floor is
 * 17, which is already linear_add()'s over the [5120,6144] family). The current NVFP4 problems
 * register the seven non-vocabulary FP8 geometries — the five parents plus the shards
 * `[17408,5120]` and `[5120,8704]` — and accept every positive T. Q8 and BF16 register the exact
 * physical N/K problems in their format-owned shape registries; the BF16 registry includes the
 * two-device shards `[7168,5120]` and `[5120,3072]`. Text/MTP problems accept every positive column
 * extent T. Registered Vision problems accept raw-patch P in
 * `{4,8,...,131072}` or merged-token V in `[1,32768]`; a matrix column does not inherently
 * represent a text token. FP32 is unsupported.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected
 * activation-compute path; none is copied into the oracle. Kernel, schedule, template instance,
 * host launcher, and T region do not create separate criteria inside one path.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. Every policy permits the existing A16 implementations of BF16 and Q4/Q5/Q6/Q8.
 * FP8 accepts all three policies; AllowA8 and AllowA4 permit the private resolver to select a
 * qualified A16 or A8 route for the registered problem and T. FP8 `[248320,5120]` retains A16
 * compute under every policy at every positive T. NVFP4 uses A16 for A16Only and
 * AllowA8; AllowA4 permits the private resolver to select either a qualified A16 route or
 * activation quantization to NVFP4 at every positive T. The selected route depends only on the
 * registered problem and T.
 *
 * @par Workspace
 * `workspace` is caller-owned call-scoped transient storage sized by
 * linear_workspace_capacity_bytes(). It must not overlap x, any weight plane, or out. Linear does
 * not allocate device memory internally.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 output matrix `[N,T]`. It must not
 * overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] workspace Caller-owned transient arena.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload admits only A16 compute and requires no transient workspace. All tensor,
 * weight, aliasing, and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 output matrix `[N,T]`.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

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
