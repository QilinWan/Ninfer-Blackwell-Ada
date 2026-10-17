#pragma once

#include "ninfer/types.h"

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product {

// Command-line forms of the tensor-parallel Engine options shared by ninfer and ninfer-serve.

namespace detail {

// A nonnegative decimal that fits int, or -1.
[[nodiscard]] inline int parse_small_decimal(std::string_view text) {
    const std::string value(text);
    if (value.empty() || value.front() == '-') { return -1; }
    errno                           = 0;
    char* end                       = nullptr;
    const unsigned long long number = std::strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != '\0' ||
        number > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        return -1;
    }
    return static_cast<int>(number);
}

} // namespace detail

// One CUDA device id.
[[nodiscard]] inline int parse_device_id(std::string_view text) {
    const int device = detail::parse_small_decimal(text);
    if (device < 0) { throw std::invalid_argument("invalid device: " + std::string(text)); }
    return device;
}

// `--tp 1|2`.
[[nodiscard]] inline int parse_tp(std::string_view text) {
    const int tp = detail::parse_small_decimal(text);
    if (tp != 1 && tp != 2) { throw std::invalid_argument("--tp must be 1 or 2"); }
    return tp;
}

// `--devices A[,B]`, rank 0 first.
[[nodiscard]] inline std::vector<int> parse_devices(std::string_view text) {
    std::vector<int> devices;
    while (true) {
        const std::size_t comma = text.find(',');
        devices.push_back(parse_device_id(text.substr(0, comma)));
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    return devices;
}

// Completes the parsed `--tp`, `--devices` and `--device` into the Engine's rank list: without
// `--devices` only tp 1 is valid and rank 0 is `--device`; otherwise the list holds exactly tp ids
// and its first one becomes the rank 0 device, which an explicit `--device` must name.
inline void resolve_tensor_parallel_devices(int tp, std::vector<int>& devices, int& device,
                                            bool device_explicit) {
    if (devices.empty()) {
        if (tp != 1) { throw std::invalid_argument("--tp 2 requires --devices A,B"); }
        devices = {device};
        return;
    }
    if (devices.size() != static_cast<std::size_t>(tp)) {
        throw std::invalid_argument("--devices must list exactly --tp device ids");
    }
    if (device_explicit && devices.front() != device) {
        throw std::invalid_argument("--device and --devices disagree on the rank 0 device");
    }
    device = devices.front();
}

// Capability gate for the tensor-parallel route: Ada keeps this interface, not an implementation.
// Both verdicts are pure functions with zero CUDA references, so a CPU test reads exactly the
// verdict the Engine gives at startup, and the Engine calls them once it owns the rank
// DeviceContexts. A non-empty return is the complete, user-facing reason. `--tp 1` is never
// restricted by these verdicts.
[[nodiscard]] inline std::optional<std::string>
tensor_parallel_capability_verdict(int cc_major, int cc_minor) noexcept {
    if (cc_major == 12 && cc_minor == 0) { return std::nullopt; }
    return std::string("--tp 2 requires sm_120: the tensor-parallel layer's shard TMA, split-K ") +
           "workspace and mailbox paths are only qualified on compute capability 12.0 (this fork's "
           "boards: 2x RTX 5060 Ti, CC 12.0). The selected devices report " +
           std::to_string(cc_major) + '.' + std::to_string(cc_minor) +
           ", where the shards have no execution route; run single-device --tp 1.";
}

// `bf16`, `int8-G64` and `nvfp4-G16` KV have a tensor-parallel route: the two-device plan registers
// KV page pools and execution tables for those layouts, and the head-local [256,12,2] geometry is
// instantiated by their attention kernels alone (TextContext, require_causal_geometry). The FP8
// families and the 4-bit `rk4v4-e8` storage (Ada's KV family) name no such route; qualifying them
// is Ada-phase work, not part of this interface.
[[nodiscard]] inline std::optional<std::string>
tensor_parallel_kv_verdict(KvCacheStorage storage) noexcept {
    if (storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Int8Group64 ||
        storage == KvCacheStorage::Nvfp4Group16) {
        return std::nullopt;
    }
    std::string_view name = "requested";
    switch (storage) {
    case KvCacheStorage::RK4V4E8:
        name = "rk4v4-e8";
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        name = "fp8";
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        name = "k8v4";
        break;
    default:
        break;
    }
    return std::string("--tp 2 supports bf16, int8 and nvfp4 KV caches: the two-device plan ") +
           "registers no execution route for the " + std::string(name) +
           " KV storage; choose --kv-dtype bf16, int8 or nvfp4, or run single-device --tp 1.";
}

} // namespace ninfer::product
