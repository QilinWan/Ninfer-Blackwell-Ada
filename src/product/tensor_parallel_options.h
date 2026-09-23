#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
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

} // namespace ninfer::product
