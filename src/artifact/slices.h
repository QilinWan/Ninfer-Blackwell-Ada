#pragma once

#include "core/weight_view.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::artifact {

// One process drives at most two devices (core/device.h ExecutionContext).
inline constexpr std::size_t kMaximumDevices = 2;

// One contiguous byte range copied from a complete encoded parent into a shard parent.
struct PlaneCopy {
    std::uint64_t source_offset = 0; // From the first encoded byte of the parent.
    std::uint64_t dest_offset   = 0; // From the first encoded byte of the shard.
    std::uint64_t bytes         = 0;

    friend bool operator==(const PlaneCopy&, const PlaneCopy&) = default;
};

// Half-open [begin, begin + count) along the sliced axis, in logical coordinates.
struct SliceRange {
    std::uint64_t begin = 0;
    std::uint64_t count = 0;

    friend bool operator==(const SliceRange&, const SliceRange&) = default;
};

// A shard is a complete parent with the same format and layout whose shape narrows one axis to
// the concatenation of the selected ranges. Every layout keeps logical values in whole bytes of
// its planes, so a shard is a list of parent byte ranges placed at the shard's own plane offsets.
// Plane alignment gaps of the shard are not covered by any copy; the layouts define them as zero.
struct TensorSlice {
    WeightGeometry geometry;
    std::vector<PlaneCopy> copies;
};

// Axis 0. Ranges are nonempty, ascending, disjoint and inside the parent. BlockScaleK16M128x4
// additionally requires 128-row tiles; every other layout accepts any row boundary.
[[nodiscard]] TensorSlice tensor_row_slice(const WeightGeometry& parent,
                                           std::span<const SliceRange> rows);

// Axis 1 of a matrix, with the ranges concatenated within every row. Only Contiguous accepts
// more than one range; the other layouts group, tile or swizzle columns. RowSplit ranges are
// multiples of 128 columns, BlockScaleK16M128x4 ranges multiples of 64, and RowScale keeps its
// per-row scale plane whole because the rows are unchanged.
[[nodiscard]] TensorSlice tensor_column_slice(const WeightGeometry& parent,
                                              std::span<const SliceRange> columns);

// Placement of one device-resident parent. Rows and Columns give each device its own ranges of
// that axis; the remaining kinds place the complete parent. PrimaryOnly holds it on device 0 and
// SingleDevice on `device`.
enum class ShardAxis : std::uint8_t {
    Replicated,
    Rows,
    Columns,
    PrimaryOnly,
    SingleDevice,
};

struct ShardPlacement {
    ShardAxis axis = ShardAxis::Replicated;
    int device     = 0; // Holder under SingleDevice.
    std::array<std::vector<SliceRange>, kMaximumDevices> device_ranges;
};

[[nodiscard]] constexpr bool is_sharded(ShardAxis axis) noexcept {
    return axis == ShardAxis::Rows || axis == ShardAxis::Columns;
}

// Rows or Columns slice selected by `axis`.
[[nodiscard]] TensorSlice tensor_slice(const WeightGeometry& parent, ShardAxis axis,
                                       std::span<const SliceRange> ranges);

} // namespace ninfer::artifact
