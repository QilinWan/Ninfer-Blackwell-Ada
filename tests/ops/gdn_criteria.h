#pragma once

// The gated_delta_net Op's registered numerical criteria, shared by its conformance suite and the
// two-device head-split suite so that a shard is judged by the contract of the geometry it splits.
//
// Both are distances to gdn_ref.h's exact FP64 recurrence over a whole output block, fitted on
// test_gated_delta_net.cpp's fixture at T <= 128. The recurrence accumulates BF16 error with its
// length, so applying them far outside that range measures the length, not the implementation.

#include "ops/op_check.h"

namespace ninfer::test {

// BF16 output of the recurrence, promoted and compared against the FP64 ideal.
constexpr ReductionCriterion gated_delta_net_output_bf16_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/5.5e-3};
}

// FP32 published state after the last token.
constexpr ReductionCriterion gated_delta_net_state_fp32_criterion() {
    return {/*relative_l2=*/2.7e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/3.9e-3};
}

} // namespace ninfer::test
