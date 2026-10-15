#pragma once

// Two-device (tp 2) form of linear_dynamic_grouped_conv_add. Our file:
// ninfer/ops/dynamic_grouped_conv.h is upstream's and carries no hook.

#include "ninfer/ops/dynamic_grouped_conv.h"

namespace ninfer::ops {

/**
 * The finish half of linear_dynamic_grouped_conv_add with the projection supplied: for H=5120,
 * G=320, h=16*g+j,
 *
 *     residual[h,i,b] += (base_kernel[h,0,1] + finish_delta[g,0,i,b]) * z[h,i,b]
 *                      + I(i>0) * (base_kernel[h,1,1] + finish_delta[g,1,i,b]) * z[h,i-1,b]
 *
 * with z = `projected`, contiguous BF16 [5120,W,B]; every other operand as in
 * linear_dynamic_grouped_conv_add (W in [2,16], B in [1,8]). It is upstream's finish kernel of the
 * materialized Q8 route (q8_dynamic_grouped_conv_add_materialized.cu), which reads the projection
 * from BF16 storage: a row-parallel drafter sums the ranks' partial projections with allreduce_sum
 * and runs this on both ranks, which then hold the identical residual. Elementwise, no workspace.
 */
void dynamic_grouped_conv_finish_add(const Tensor& projected, const Tensor& base_kernel,
                                     const Tensor& finish_delta, Tensor& residual,
                                     cudaStream_t stream);

} // namespace ninfer::ops
