#pragma once

// Two-device (tp 2) forms of gdn_gating_proj. Our file: ninfer/ops/gdn_gating_proj.h is upstream's and carries no hook.

#include "ninfer/ops/gdn_gating_proj.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Tensor-parallel form over two devices.
//
// Rank r owns value heads [24r,24r+24): rows [24r,24r+24) of the A and B weights, A_log and
// dt_bias, and writes those heads' g and beta. Value head h belongs to key head h / 3 in the GDN
// core, so each rank's 24 value heads are exactly the 8 complete key-head groups its GDN input
// projection shard owns and the core needs nothing from the peer. `x` holds the same activation
// on both ranks; nothing is communicated.
//
// Every requirement of the single-device form applies per rank at 24 heads. `x[r]`, the rank's
// weights and vectors, `ws[r]`, `g[r]` and `beta[r]` must be resident on `ec.dev[r]`, and rank r's
// work is enqueued on `ec.dev[r]->stream`. Inputs staged on a device's legacy default stream must
// be retired before the call; the call does not synchronize and preserves the current device.

/// Per-rank transient capacity at the shard profile (`heads` 24, `input_rows` 5120).
[[nodiscard]] std::size_t gdn_gating_proj_column_parallel_workspace_capacity_bytes(
    std::int32_t heads, std::int32_t input_rows, std::int32_t min_tokens, std::int32_t max_tokens);

/// Two-weight form: per-rank BF16 `a_weight`/`b_weight` [24,5120], FP32 `g`/`beta` [24,T].
void gdn_gating_proj_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& a_weight,
    const std::array<Weight, 2>& b_weight, const std::array<Tensor, 2>& A_log,
    const std::array<Tensor, 2>& dt_bias, const std::array<WorkspaceArena*, 2>& ws,
    const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta, const ExecutionContext& ec);

/// Contiguous-parent form: per-rank BF16 `ab_weight` [48,5120] holding the rank's A rows in
/// [0,24) and its B rows in [24,48).
void gdn_gating_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& ab_weight,
                                     const std::array<Tensor, 2>& A_log,
                                     const std::array<Tensor, 2>& dt_bias,
                                     const std::array<WorkspaceArena*, 2>& ws,
                                     const std::array<Tensor, 2>& g,
                                     const std::array<Tensor, 2>& beta, const ExecutionContext& ec);

} // namespace ninfer::ops
