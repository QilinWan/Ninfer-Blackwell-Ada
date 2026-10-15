#pragma once

// Per-GPU tuning of this fork: the one place where a board gets its measured values. Our file
// (header-only, so it needs no registration in upstream's CMake lists).
//
// Rows are matched by device properties (compute capability, SM count), first match wins; the
// last row matches every device and carries the values measured on two RTX 5070 Ti, which is what
// every device ran before the table existed. A new GPU adds a row above it (docs/maintainer/
// upstream-merge.md, "Adding a new GPU"); nothing else is edited for its values.
//
// Per-GPU choices that are NOT in the table, because they are compile-time schedule selections
// inside upstream's Op code shared by every device (a runtime switch would change upstream's
// selectors; they run on every GPU as measured on the RTX 5070 Ti):
//   - NVFP4 A16 sliced-K staged-token/row-tile instances and the MLP-down SIMT MinBlocksPerSm
//     (91582814: nvfp4_schedule.cuh, nvfp4_linear_swiglu_small_t.cu, linear shapes n5120_k17408);
//   - the INT8 attention split rule itself (169514ea, 21ad2f4a: int8/plan.cpp, grouped_mma.cuh),
//     which reads the SM count below;
//   - the mailbox poller's spin limit kPeerSpinLimit (ops/kernel/peer_exchange.cuh).
// The shard and half problems otherwise inherit upstream's schedules and crossovers (A16/A8/A4
// thresholds, MMA bands, TMA tiles) from the problem they halve.

#include "core/device.h" // CUDA_CHECK

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ninfer::tp2 {

struct DeviceTuning {
    // Where the values come from (board, clocks, date, commit).
    const char* source;
    // Match keys; 0 matches any value.
    int compute_capability; // major * 10 + minor: 120 for sm_120
    int sm_count;

    // Causal attention (every width): SM count the INT8 plan sizes one wave of splits for;
    // 0 = the device's own SM count (ops/softmax_attention/common/causal_partition.cpp).
    int attention_sm_count;

    // NVFP4 attn_input_proj [7168,5120] shard, A4 TMA route: the first T that reads 256-token
    // scale tiles; below it the 128-token tiles (upstream and the parent switch at T=1024).
    // ops/attn_input_proj/tp2/nvfp4_attn_input_shard_a4.cu.
    std::int32_t nvfp4_attn_input_shard_first_tiled256_tokens;

    // CUDA Graph memory allowance per device at tp 2 (program/planning/startup.cpp):
    // max(3 x observed, 8 MiB), observed = the free memory prepare_graphs() consumed per rank at
    // 32K context, concurrency 1, INT8 KV; scaled by max concurrency.
    std::size_t tp2_ordinary_graph_allowance_bytes;      // per batch size (one class)
    std::size_t tp2_mtp_graph_class_allowance_bytes;     // per topology class and batch size
    std::size_t tp2_dflash2_graph_class_allowance_bytes; // per topology class and batch size
};

inline constexpr std::size_t kTuningMiB = 1024ULL * 1024ULL;

// First match wins; keep the catch-all row last.
inline constexpr std::array<DeviceTuning, 2> kDeviceTuningRows{{
    {"RTX 5070 Ti (sm_120, 70 SMs), 2x on PCIe 5.0 x8 without P2P, 2.08 GHz, 2026-09: "
     "attn shard tile 2a596191; graph allowances measured ordinary 2.0/2.0 MiB, MTP3 2.0/2.0 MiB, "
     "DFlash2 K=4 18.0/12.0 MiB over five classes (rank 0/1)",
     120, 70, 0, 1025, 8 * kTuningMiB, 8 * kTuningMiB, 11 * kTuningMiB},
    {"any other device: the RTX 5070 Ti values, not re-measured", 0, 0, 0, 1025, 8 * kTuningMiB,
     8 * kTuningMiB, 11 * kTuningMiB},
}};
static_assert(kDeviceTuningRows.back().compute_capability == 0 &&
              kDeviceTuningRows.back().sm_count == 0);

namespace detail {

inline const DeviceTuning& match_device_tuning(int compute_capability, int sm_count) {
    for (const DeviceTuning& row : kDeviceTuningRows) {
        if ((row.compute_capability == 0 || row.compute_capability == compute_capability) &&
            (row.sm_count == 0 || row.sm_count == sm_count)) {
            return row;
        }
    }
    return kDeviceTuningRows.back();
}

inline constexpr int kDeviceTuningMaxDevices = 64;

// Row index plus one per device; 0 = not resolved yet. Concurrent first queries store the same
// value.
inline std::array<std::atomic<int>, kDeviceTuningMaxDevices>& device_tuning_cache() {
    static std::array<std::atomic<int>, kDeviceTuningMaxDevices> rows{};
    return rows;
}

} // namespace detail

// The row of `device`, resolved once per device.
inline const DeviceTuning& device_tuning(int device) {
    const bool tracked = device >= 0 && device < detail::kDeviceTuningMaxDevices;
    if (tracked) {
        const int cached = detail::device_tuning_cache()[static_cast<std::size_t>(device)].load(
            std::memory_order_relaxed);
        if (cached > 0) return kDeviceTuningRows[static_cast<std::size_t>(cached - 1)];
    }
    int major = 0;
    int minor = 0;
    int sms   = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const DeviceTuning& row = detail::match_device_tuning(major * 10 + minor, sms);
    if (tracked) {
        detail::device_tuning_cache()[static_cast<std::size_t>(device)].store(
            static_cast<int>(&row - kDeviceTuningRows.data()) + 1, std::memory_order_relaxed);
    }
    return row;
}

// The row of the current CUDA device.
inline const DeviceTuning& current_device_tuning() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device_tuning(device);
}

} // namespace ninfer::tp2
