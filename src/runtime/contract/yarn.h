#pragma once

#include "ninfer/types.h"

#include <bit>
#include <cmath>
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

// A saved KV state may only be restored under the positional transform that created it.
// Native keys remain unchanged; the algorithm version also guards future formula changes.
inline std::string yarn_cache_binding(const YarnOptions& yarn) {
    if (yarn.factor == 1.0F) { return {}; }
    return "\nyarn-hf-v1:" + std::to_string(std::bit_cast<std::uint32_t>(yarn.factor)) + ":" +
           std::to_string(yarn.original_context);
}

} // namespace ninfer::runtime
