#pragma once

#include "ninfer/types.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {

inline void validate_yarn(const YarnOptions& yarn, std::uint32_t max_context) {
    if (!std::isfinite(yarn.factor) || yarn.factor < 1.0F || yarn.factor > 4.0F) {
        throw std::invalid_argument("--rope-yarn-factor must be finite and in [1,4]");
    }
    if (yarn.original_context != 262144) {
        throw std::invalid_argument(
            "--rope-original-max-position must be the Qwen3.8 native window (262144)");
    }
    if (yarn.factor > 1.0F && static_cast<double>(max_context) >
                                  static_cast<double>(yarn.original_context) * yarn.factor) {
        throw std::invalid_argument("--max-context exceeds the YaRN-scaled reference window");
    }
}

// The YaRN coefficient table expresses exactly one rotary domain: the 32-entry inverse_frequency
// array of ops/rope.h is the half of a 64-wide rotary on a 256-wide head. Every other registered
// rotary domain — the D128/R128 full-head text route and the D72/R72 vision route — rotates natively
// by contract, so a factor requested for one of them is a configuration error, not a transform the
// kernel owning that domain silently drops. The model layer enforces this against the loaded
// artifact, which is where the rotary shape is known.
inline void validate_yarn_rotary_domain(std::uint32_t head_dim, std::uint32_t rotary_dim,
                                        float factor) {
    if (factor <= 1.0F) { return; }
    if (head_dim != 256U || rotary_dim != 64U) {
        throw std::invalid_argument(
            "--rope-yarn-factor supports the D256/R64 text rotary domain only; this artifact is D" +
            std::to_string(head_dim) + "/R" + std::to_string(rotary_dim) +
            ", whose RoPE stays native (include/ninfer/ops/rope.h)");
    }
}

// A saved KV state may only be restored under the positional transform that created it.
// Native keys remain unchanged; the algorithm version also guards future formula changes.
inline std::string yarn_cache_binding(const YarnOptions& yarn) {
    if (yarn.factor == 1.0F) { return {}; }
    return "\nyarn-hf-v1:" + std::to_string(std::bit_cast<std::uint32_t>(yarn.factor)) + ":" +
           std::to_string(yarn.original_context);
}

} // namespace ninfer::runtime
