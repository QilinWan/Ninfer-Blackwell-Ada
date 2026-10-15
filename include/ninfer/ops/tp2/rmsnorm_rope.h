#pragma once

// Two-device (tp 2) form of rmsnorm_rope. Our file: ninfer/ops/rmsnorm_rope.h is upstream's and
// carries no hook.

#include "ninfer/ops/rmsnorm_rope.h"

namespace ninfer::ops {

/**
 * The pair form of rmsnorm_rope over one tensor-parallel rank's contiguous block of heads: q BF16
 * [128,16,W,B] and k BF16 [128,4,W,B], every other contract as the [128,32]/[128,8] pair. Each
 * head is normalized and rotated by the same per-head code as the complete profile, so rank r's
 * heads equal the complete profile's heads [16 r, 16 r + 16) and [4 r, 4 r + 4) bit for bit.
 */
void rmsnorm_rope_head_block(const Tensor& positions, const Tensor& q_norm_weight,
                             const Tensor& k_norm_weight, Tensor& q, Tensor& k,
                             cudaStream_t stream);

} // namespace ninfer::ops
