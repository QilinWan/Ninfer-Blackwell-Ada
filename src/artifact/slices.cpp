#include "artifact/slices.h"

#include "artifact/schema.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::artifact {
namespace {

// BlockScaleK16M128x4: the scale plane is ordered by 128-row tile, then by 64-column tile of
// 512 bytes (core/weight_view.cpp weight_scale_offset).
constexpr std::uint64_t kScaleTileRows    = 128;
constexpr std::uint64_t kScaleTileColumns = 64;
constexpr std::uint64_t kScaleTileBytes   = 512;
constexpr std::uint64_t kRowSplitColumns  = 128;
constexpr std::uint64_t kDivisorBytes     = 4;

void require_slice(bool condition, std::string_view reason) {
    if (!condition) { throw ArtifactError("tensor slice: " + std::string(reason)); }
}

std::uint64_t selected_count(std::span<const SliceRange> ranges, std::uint64_t extent,
                             std::uint64_t alignment) {
    require_slice(!ranges.empty(), "at least one range is required");
    std::uint64_t total        = 0;
    std::uint64_t previous_end = 0;
    for (const auto& range : ranges) {
        require_slice(range.count != 0, "ranges must be nonempty");
        require_slice(range.begin >= previous_end, "ranges must be ascending and disjoint");
        require_slice(range.begin <= extent && extent - range.begin >= range.count,
                      "range exceeds the parent axis");
        require_slice(range.begin % alignment == 0 && range.count % alignment == 0,
                      "range is not aligned to the layout tile");
        previous_end = range.begin + range.count;
        total += range.count;
    }
    return total;
}

WeightGeometry narrowed(const WeightGeometry& parent, std::size_t axis, std::uint64_t extent) {
    auto shape  = parent.shape;
    shape[axis] = extent;
    try {
        return weight_geometry(parent.format, parent.layout, shape);
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("tensor slice: ") + error.what());
    }
}

// Appends a copy, extending the previous one when both source and destination continue it.
void push(std::vector<PlaneCopy>& copies, std::uint64_t source, std::uint64_t destination,
          std::uint64_t bytes) {
    if (!bytes) { return; }
    if (!copies.empty()) {
        auto& last = copies.back();
        if (last.source_offset + last.bytes == source &&
            last.dest_offset + last.bytes == destination) {
            last.bytes += bytes;
            return;
        }
    }
    copies.push_back({source, destination, bytes});
}

// A row-addressable plane: `unit_bytes` per `unit_rows` rows, in parent and shard alike.
struct RowPlane {
    std::uint64_t source_base = 0;
    std::uint64_t dest_base   = 0;
    std::uint64_t unit_bytes  = 0;
    std::uint64_t unit_rows   = 1;
};

void copy_rows(std::vector<PlaneCopy>& copies, const RowPlane& plane,
               std::span<const SliceRange> rows) {
    std::uint64_t destination = 0;
    for (const auto& range : rows) {
        const auto units = range.count / plane.unit_rows;
        push(copies, plane.source_base + range.begin / plane.unit_rows * plane.unit_bytes,
             plane.dest_base + destination * plane.unit_bytes, units * plane.unit_bytes);
        destination += units;
    }
}

// A plane whose rows narrow: each of `rows` source rows of `source_stride` bytes contributes
// `bytes` from `source_skip` to the start of its shard row of `dest_stride` bytes.
struct ColumnPlane {
    std::uint64_t source_base   = 0;
    std::uint64_t dest_base     = 0;
    std::uint64_t source_stride = 0;
    std::uint64_t dest_stride   = 0;
};

void copy_columns(std::vector<PlaneCopy>& copies, const ColumnPlane& plane, std::uint64_t rows,
                  std::uint64_t source_skip, std::uint64_t bytes) {
    for (std::uint64_t row = 0; row < rows; ++row) {
        push(copies, plane.source_base + row * plane.source_stride + source_skip,
             plane.dest_base + row * plane.dest_stride, bytes);
    }
}

void copy_divisor(std::vector<PlaneCopy>& copies, const WeightGeometry& parent,
                  const WeightGeometry& shard) {
    // The NVFP4 weight divisor scales the whole matrix, so every shard carries it unchanged.
    if (parent.format == QType::NVFP4) {
        push(copies, parent.divisor_offset, shard.divisor_offset, kDivisorBytes);
    }
}

} // namespace

TensorSlice tensor_row_slice(const WeightGeometry& parent, std::span<const SliceRange> rows) {
    require_slice(!parent.shape.empty(), "a row slice requires a tensor of rank one or more");
    const bool tiled = parent.layout == QuantLayout::BlockScaleK16M128x4;
    const auto count = selected_count(rows, parent.shape[0], tiled ? kScaleTileRows : 1);
    TensorSlice out{narrowed(parent, 0, count), {}};
    const auto& shard = out.geometry;
    switch (parent.layout) {
    case QuantLayout::Contiguous:
        copy_rows(out.copies, {0, 0, parent.bytes / parent.shape[0]}, rows);
        break;
    case QuantLayout::RowSplit:
        copy_rows(out.copies, {0, 0, parent.code_bytes_per_row}, rows);
        if (parent.high_bytes) {
            copy_rows(out.copies,
                      {parent.high_offset, shard.high_offset, parent.high_bytes_per_row}, rows);
        }
        copy_rows(out.copies, {parent.scale_offset, shard.scale_offset, parent.scale_bytes_per_row},
                  rows);
        break;
    case QuantLayout::RowScale:
        copy_rows(out.copies, {0, 0, parent.code_bytes_per_row}, rows);
        copy_rows(out.copies, {parent.scale_offset, shard.scale_offset, parent.scale_bytes_per_row},
                  rows);
        break;
    case QuantLayout::BlockScaleK16M128x4:
        copy_rows(out.copies, {0, 0, parent.code_bytes_per_row}, rows);
        copy_rows(out.copies,
                  {parent.scale_offset, shard.scale_offset,
                   kScaleTileRows * parent.scale_bytes_per_row, kScaleTileRows},
                  rows);
        copy_divisor(out.copies, parent, shard);
        break;
    default:
        throw ArtifactError("tensor slice: unknown layout");
    }
    return out;
}

TensorSlice tensor_column_slice(const WeightGeometry& parent, std::span<const SliceRange> columns) {
    require_slice(parent.shape.size() == 2, "a column slice requires a matrix");
    const auto rows      = parent.shape[0];
    const auto alignment = parent.layout == QuantLayout::RowSplit              ? kRowSplitColumns
                           : parent.layout == QuantLayout::BlockScaleK16M128x4 ? kScaleTileColumns
                                                                               : 1;
    const auto count     = selected_count(columns, parent.shape[1], alignment);
    require_slice(parent.layout == QuantLayout::Contiguous || columns.size() == 1,
                  "only a contiguous layout accepts several column ranges");
    TensorSlice out{narrowed(parent, 1, count), {}};
    const auto& shard  = out.geometry;
    const auto& column = columns.front();
    switch (parent.layout) {
    case QuantLayout::Contiguous: {
        const auto word = parent.code_bytes_per_row / parent.shape[1];
        for (std::uint64_t row = 0; row < rows; ++row) {
            auto destination = row * shard.code_bytes_per_row;
            for (const auto& range : columns) {
                push(out.copies, row * parent.code_bytes_per_row + range.begin * word, destination,
                     range.count * word);
                destination += range.count * word;
            }
        }
        break;
    }
    case QuantLayout::RowSplit: {
        // 128-column ranges keep whole groups and give the shard a padded width equal to its
        // own column count, so none of the parent's trailing padding groups is carried over.
        const auto groups = parent.padded_columns / parent.group_size;
        const auto skip   = column.begin / parent.group_size;
        const auto taken  = column.count / parent.group_size;
        const auto code   = parent.code_bytes_per_row / groups;
        const auto high   = parent.high_bytes_per_row / groups;
        const auto scale  = parent.scale_bytes_per_row / groups;
        copy_columns(out.copies, {0, 0, parent.code_bytes_per_row, shard.code_bytes_per_row}, rows,
                     skip * code, taken * code);
        if (high) {
            copy_columns(out.copies,
                         {parent.high_offset, shard.high_offset, parent.high_bytes_per_row,
                          shard.high_bytes_per_row},
                         rows, skip * high, taken * high);
        }
        copy_columns(out.copies,
                     {parent.scale_offset, shard.scale_offset, parent.scale_bytes_per_row,
                      shard.scale_bytes_per_row},
                     rows, skip * scale, taken * scale);
        break;
    }
    case QuantLayout::RowScale:
        // Each partial product is scaled by the same row multiplier, so the plane stays whole.
        copy_columns(out.copies, {0, 0, parent.code_bytes_per_row, shard.code_bytes_per_row}, rows,
                     column.begin, column.count);
        push(out.copies, parent.scale_offset, shard.scale_offset, parent.scale_bytes);
        break;
    case QuantLayout::BlockScaleK16M128x4: {
        copy_columns(out.copies, {0, 0, parent.code_bytes_per_row, shard.code_bytes_per_row}, rows,
                     column.begin / 2, column.count / 2);
        // Within a 128-row tile the selected 64-column scale tiles are contiguous.
        const auto parent_tiles = parent.shape[1] / kScaleTileColumns;
        const auto shard_tiles  = column.count / kScaleTileColumns;
        copy_columns(out.copies,
                     {parent.scale_offset, shard.scale_offset, parent_tiles * kScaleTileBytes,
                      shard_tiles * kScaleTileBytes},
                     rows / kScaleTileRows, column.begin / kScaleTileColumns * kScaleTileBytes,
                     shard_tiles * kScaleTileBytes);
        copy_divisor(out.copies, parent, shard);
        break;
    }
    default:
        throw ArtifactError("tensor slice: unknown layout");
    }
    return out;
}

TensorSlice tensor_slice(const WeightGeometry& parent, ShardAxis axis,
                         std::span<const SliceRange> ranges) {
    if (axis == ShardAxis::Rows) { return tensor_row_slice(parent, ranges); }
    if (axis == ShardAxis::Columns) { return tensor_column_slice(parent, ranges); }
    throw ArtifactError("tensor slice: placement does not narrow an axis");
}

} // namespace ninfer::artifact
