#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

#include <vector>

namespace ninfer::ops::detail {

// Capture reserves partials for the largest live row. Producer and merge use
// the same live count; a wider capture never changes a row's work partition.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    int capacity                    = 1;
    int target                      = 1;
    int key_shift                   = 6; // log2 of the minimum KV keys per split
    // A balanced partition gives a row of more than `target` units the fewest splits that keep
    // its longest split as short as `target` splits would: ceil(units / ceil(units / target)).
    bool balanced = false;

    __host__ __device__ int units(int visible) const {
        return (visible + (1 << key_shift) - 1) >> key_shift;
    }
    // Largest live count over rows of up to `visible` keys: the grid and partials capacity.
    __host__ __device__ int bound(int visible) const {
        const int count = units(visible);
        return count < target ? count : target;
    }
    // Units per split of the longest split, 1 while the row has no more units than `target`.
    __host__ __device__ unsigned longest(int visible) const {
        const unsigned count = static_cast<unsigned>(units(visible));
        const unsigned limit = static_cast<unsigned>(target);
        return count <= limit ? 1u : (count + limit - 1u) / limit;
    }
    __host__ __device__ int active(int visible) const {
        const int count = units(visible);
        if (count <= target) return count;
        if (!balanced) return target;
        const unsigned run = longest(visible);
        return static_cast<int>((static_cast<unsigned>(count) + run - 1u) / run);
    }
};

// Complete groups of independent query tiles should fill at least 90% of the SMs.
// Each dtype chooses whether a shortfall warrants a larger CTA budget.
inline constexpr bool causal_query_tiles_underfill_sms(int independent_tiles,
                                                       int multiprocessor_count) {
    const std::int64_t filled =
        (multiprocessor_count / independent_tiles) * static_cast<std::int64_t>(independent_tiles);
    return filled < static_cast<std::int64_t>(multiprocessor_count) * 9 / 10;
}

inline constexpr int causal_partition_target(std::int64_t cta_budget, int independent_tiles) {
    return static_cast<int>(
        std::clamp<std::int64_t>(cta_budget / independent_tiles, 1, CausalKvPartition::kMaxSplits));
}

} // namespace ninfer::ops::detail
