#pragma once

#include "ninfer/types.h"
#include "product/tensor_parallel_options.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product {

// Command-line forms of the Vision Engine options shared by ninfer and ninfer-serve.

// `--max-vision-tokens N`, the merged-token ceiling of one image or video item.
[[nodiscard]] inline std::uint32_t parse_max_vision_tokens(std::string_view text) {
    const int tokens = detail::parse_small_decimal(text);
    if (tokens < static_cast<int>(kMinimumMaxVisionTokens) ||
        tokens > static_cast<int>(kMaximumMaxVisionTokens)) {
        throw std::invalid_argument("--max-vision-tokens must be in [" +
                                    std::to_string(kMinimumMaxVisionTokens) + "," +
                                    std::to_string(kMaximumMaxVisionTokens) + "]");
    }
    return static_cast<std::uint32_t>(tokens);
}

// Checks `--vision-device` and `--max-vision-tokens` after resolve_tensor_parallel_devices: both
// need `--vision`, and the Vision tower's device must be one of the ranks' devices.
inline void validate_vision_options(bool enable_vision, const std::optional<int>& vision_device,
                                    const std::optional<std::uint32_t>& max_vision_tokens,
                                    const std::vector<int>& devices) {
    if ((vision_device || max_vision_tokens) && !enable_vision) {
        throw std::invalid_argument("--vision-device and --max-vision-tokens require --vision");
    }
    if (vision_device &&
        std::find(devices.begin(), devices.end(), *vision_device) == devices.end()) {
        throw std::invalid_argument("--vision-device must be one of the --devices ids");
    }
}

} // namespace ninfer::product
