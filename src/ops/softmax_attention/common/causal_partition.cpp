#include "ops/softmax_attention/common/causal_partition.h"

#include "core/device.h" // CUDA_CHECK
#include "core/tp2/device_tuning.h"

#include <algorithm>
#include <array>
#include <atomic>

namespace ninfer::ops::detail {
namespace {

constexpr int kMaxDevices = 64;

// 0 = not queried yet. Concurrent first queries store the same value.
std::array<std::atomic<int>, kMaxDevices> g_sm_counts{};

// The device's SM count, or the per-GPU override of core/tp2/device_tuning.h.
int device_sm_count(int device) {
    const bool tracked = device >= 0 && device < kMaxDevices;
    if (tracked) {
        const int cached = g_sm_counts[device].load(std::memory_order_relaxed);
        if (cached > 0) return cached;
    }
    int count = tp2::device_tuning(device).attention_sm_count;
    if (count <= 0) {
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    }
    if (tracked) g_sm_counts[device].store(count, std::memory_order_relaxed);
    return count;
}

} // namespace

int causal_attention_sm_count() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device_sm_count(device);
}

const std::vector<int>& causal_attention_device_sm_counts() {
    static const std::vector<int> counts = [] {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
            (void)cudaGetLastError(); // no visible device: plan for the reference target
            return std::vector<int>{kCausalAttentionSmCount};
        }
        std::vector<int> result;
        for (int device = 0; device < devices; ++device) result.push_back(device_sm_count(device));
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }();
    return counts;
}

} // namespace ninfer::ops::detail
