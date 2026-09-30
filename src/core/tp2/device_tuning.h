#pragma once

// Per-GPU tuning of the two-device layer: the choices that were measured on one board and that
// another board may want differently, in one table keyed by device properties. Our file.
//
// Rows match on (compute capability, SM count); the first match wins, and the last row matches
// every device. A new GPU adds a row here - never an edit inside a launcher. Values are measured
// numbers, so a row states where its numbers came from; an unmeasured placeholder says so.
//
// This tree runs 2x RTX 5060 Ti 16 GB (sm_120, CC 12.0, 36 SM, no peer access). The user's ruling
// is that upstream's 76-86 / 106 tok/s figures are reference numbers only: every capability claim
// is settled by measurement on this board, so a placeholder row keeps this tree's existing behavior
// until the D-window measures replace it.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::core {

struct DeviceTuning {
    // Where the values come from (board, date, and the source that carried them).
    const char* source;
    // Match keys; 0 matches any value.
    int compute_capability; // major * 10 + minor, so 120 for sm_120
    int sm_count;

    // NVFP4 attn_input_proj [7168,5120] shard, A4 TMA route: the first T that reads 256-token
    // scale tiles. 1025 is the RTX 5070 Ti measurement (the donor's 2a596191: its 70 SM gives the
    // shard's 56 row tiles 448 CTAs at the 128-token tile, 224 at 256); 1024 means "no exception",
    // i.e. the shard follows its parent. 36 SM has not been measured here, so the 5060 Ti row
    // keeps 1024 until the D4 window.
    std::int32_t nvfp4_attn_input_shard_first_tiled256_tokens;

    // CUDA Graph driver allowance per graph class, in bytes, for the resident lanes. Placeholders
    // are the donor's 24 MiB per class; S7/D2 calibrates the three tiers on this board (the
    // launcher reads no field here yet, and adding a read without the measurement is the same
    // unmeasured-change risk the 130/142 steps carry).
    std::int64_t graph_allowance_ordinary_bytes;
    std::int64_t graph_allowance_mtp_bytes;
    std::int64_t graph_allowance_dflash2_bytes;
};

inline constexpr DeviceTuning kDeviceTuningTable[]{
    // 2x RTX 5060 Ti 16 GB: keys measured on this machine (36 SM read from the driver); the shard
    // bound and the allowance tiers are NOT measured here, so they keep the pre-table behavior.
    DeviceTuning{"2x RTX 5060 Ti 16 GB (sm_120, CC 12.0); shard TMA bound and graph allowances "
                 "unmeasured - placeholders",
                 120, 36, 1024, 24LL * 1024 * 1024, 24LL * 1024 * 1024, 24LL * 1024 * 1024},
    // Catch-all: what every device ran before this table existed (the donor's 5070 Ti numbers).
    DeviceTuning{"RTX 5070 Ti measurements carried by upstream 2a596191; default row",
                 0, 0, 1025, 24LL * 1024 * 1024, 24LL * 1024 * 1024, 24LL * 1024 * 1024},
};

// The row of the current CUDA device: queried once per device and cached. A driver failure falls
// back to the catch-all row, which is the behavior this tree had before the table.
inline const DeviceTuning& current_device_tuning() {
    static const DeviceTuning* row = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return &kDeviceTuningTable[1]; }
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) { return &kDeviceTuningTable[1]; }
        const int capability = properties.major * 10 + properties.minor;
        for (const DeviceTuning& candidate : kDeviceTuningTable) {
            if ((candidate.compute_capability == 0 ||
                 candidate.compute_capability == capability) &&
                (candidate.sm_count == 0 || candidate.sm_count == properties.multiProcessorCount)) {
                return &candidate;
            }
        }
        return &kDeviceTuningTable[1];
    }();
    return *row;
}

} // namespace ninfer::core
