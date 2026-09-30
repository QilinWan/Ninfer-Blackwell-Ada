#include "ops/tp2/device_tuning.h"

#include "core/device.h" // CUDA_CHECK

#include <array>
#include <atomic>
#include <cstddef>

namespace ninfer::ops::tp2 {
namespace {

// First match wins; keep the catch-all row last.
constexpr std::array<DeviceTuning, 2> kRows{{
    {"RTX 5070 Ti (sm_120, 70 SMs), measured 2026-09-27 at 2.08 GHz (tp2/tune-neutral 2a596191)",
     120, 70, 1025},
    {"any other device: the RTX 5070 Ti values, not re-measured", 0, 0, 1025},
}};
static_assert(kRows.back().compute_capability == 0 && kRows.back().sm_count == 0);

const DeviceTuning& match(int compute_capability, int sm_count) {
    for (const DeviceTuning& row : kRows) {
        if ((row.compute_capability == 0 || row.compute_capability == compute_capability) &&
            (row.sm_count == 0 || row.sm_count == sm_count)) {
            return row;
        }
    }
    return kRows.back();
}

constexpr int kMaxDevices = 64;
// Index into kRows plus one; 0 = not resolved yet. Concurrent first queries store the same value.
std::array<std::atomic<int>, kMaxDevices> g_rows{};

} // namespace

const DeviceTuning& device_tuning(int device) {
    const bool tracked = device >= 0 && device < kMaxDevices;
    if (tracked) {
        const int cached = g_rows[static_cast<std::size_t>(device)].load(std::memory_order_relaxed);
        if (cached > 0) return kRows[static_cast<std::size_t>(cached - 1)];
    }
    int major = 0;
    int minor = 0;
    int sms   = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const DeviceTuning& row = match(major * 10 + minor, sms);
    if (tracked) {
        g_rows[static_cast<std::size_t>(device)].store(static_cast<int>(&row - kRows.data()) + 1,
                                                       std::memory_order_relaxed);
    }
    return row;
}

const DeviceTuning& current_device_tuning() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device_tuning(device);
}

} // namespace ninfer::ops::tp2
