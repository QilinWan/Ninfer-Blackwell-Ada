#pragma once

// The gated_delta_net Op's registered numerical criteria, shared by its conformance suite and the
// two-device head-split suite so that a shard is judged by the contract of the geometry it splits.
//
// All are distances to gdn_ref.h's exact FP64 recurrence over a whole output block, fitted on
// test_gated_delta_net.cpp's fixture. The recurrence accumulates BF16 error with its length, so
// applying them far outside that fixture's range measures the length, not the implementation.
// Since upstream 0784e76f the chunked prefill route (T >= 16) is qualified per precision profile.

#include "ops/op_check.h"

#include <array>
#include <cstddef>

namespace ninfer::test {

enum class PrecisionProfile { Recurrent, ChunkedNormalized, ChunkedRaw };

struct Criteria {
    const char* name;
    ReductionCriterion out;
    ReductionCriterion state;
};

// Engineering limits for the represented public inputs, not universal forward-error
// bounds. The chunked profile keeps the master state in FP32, uses TF32 residual products and
// BF16 readout/update operands, and optionally materializes normalized Q/K once in BF16.
// Output has an additional final BF16 conversion. These profile-wide regression limits retain
// margin around the qualified error envelope and apply uniformly across head counts, sequence
// lengths and gate values. The unchanged recurrent profile retains its limits.
constexpr std::array<Criteria, 3> kCriteria{{
    {"recurrent", {4.1e-3, 5.0e-6, 5.5e-3}, {2.7e-3, 1.0e-5, 3.9e-3}},
    {"chunked-normalized", {6.0e-3, 5.0e-6, 8.0e-3}, {4.5e-3, 1.0e-5, 9.0e-3}},
    {"chunked-raw", {4.5e-3, 5.0e-6, 7.0e-3}, {3.0e-3, 1.0e-5, 4.0e-3}},
}};

constexpr PrecisionProfile prefill_profile(int tokens, bool normalize_qk) {
    // Qualify the current public prefill boundary explicitly, including T=15/16/17.
    if (tokens < 16) return PrecisionProfile::Recurrent;
    return normalize_qk ? PrecisionProfile::ChunkedNormalized : PrecisionProfile::ChunkedRaw;
}

// BF16 output of the recurrent route, promoted and compared against the FP64 ideal.
constexpr ReductionCriterion gated_delta_net_output_bf16_criterion() {
    return kCriteria[static_cast<std::size_t>(PrecisionProfile::Recurrent)].out;
}

// FP32 published state of the recurrent route after the last token.
constexpr ReductionCriterion gated_delta_net_state_fp32_criterion() {
    return kCriteria[static_cast<std::size_t>(PrecisionProfile::Recurrent)].state;
}

// Criteria of the route a prefill of `tokens` columns takes.
constexpr const Criteria& gated_delta_net_prefill_criteria(int tokens, bool normalize_qk) {
    return kCriteria[static_cast<std::size_t>(prefill_profile(tokens, normalize_qk))];
}

} // namespace ninfer::test
