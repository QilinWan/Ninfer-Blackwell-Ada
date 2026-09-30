#pragma once

// Per-GPU tuning of the two-device layer: the choices that were measured on one board and that
// another board may want different, in one table keyed by the device's properties. Our file.
//
// What is per-device today, and where it lives:
//   - the SM count the INT8 attention plan splits for: read from the device at launch
//     (ops/softmax_attention/common/causal_partition.cpp), no table entry needed;
//   - the fields of DeviceTuning below, read by our tp2 code.
// The shard and half shapes otherwise inherit upstream's schedules and crossovers (A16/A8/A4
// thresholds, MMA bands, TMA tiles) of the problem they halve, measured by upstream on an
// RTX 5090. docs/maintainer/tensor-parallel.md (Schedules and thresholds are per GPU) gives the
// procedure for another board; a measured value becomes a field here plus one read in tp2 code.
//
// The last row matches every device and carries the values measured on the RTX 5070 Ti, which is
// what every device ran before this table existed. A new row must be matched by the properties
// that make it different (compute capability, SM count), never by name.

#include <cstdint>

namespace ninfer::ops::tp2 {

struct DeviceTuning {
    // Where the values come from (board, date, commit).
    const char* source;
    // Match keys; 0 matches any value.
    int compute_capability; // major * 10 + minor, 120 for sm_120
    int sm_count;

    // NVFP4 attn_input_proj [7168,5120] shard, A4 TMA route: the first T that reads 256-token
    // scale tiles; below it the 128-token tiles (upstream and the parent switch at 1024).
    std::int32_t nvfp4_attn_input_shard_first_tiled256_tokens;
};

// The row of `device`, queried once per device and cached.
[[nodiscard]] const DeviceTuning& device_tuning(int device);
// The row of the current CUDA device.
[[nodiscard]] const DeviceTuning& current_device_tuning();

} // namespace ninfer::ops::tp2
