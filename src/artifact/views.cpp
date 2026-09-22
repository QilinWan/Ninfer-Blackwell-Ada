#include "artifact/views.h"

#include <algorithm>
#include <optional>
#include <string>

namespace ninfer::artifact {
namespace {

std::uint64_t trailing_elements(std::span<const std::uint64_t> shape) {
    return shape.size() > 1 ? weight_element_count(shape.subspan(1)) : 1;
}

// Appends the part of parent elements [begin, end) that this device's shard holds, addressed in
// the shard, and returns its element count.
std::uint64_t map_shard_part(WeightView& out, const WeightParent& parent, const DeviceShard& shard,
                             std::uint64_t begin, std::uint64_t end, const std::string& name) {
    if (shard.axis == ShardAxis::Rows) {
        // Row ranges are consecutive element ranges of the parent and of the shard.
        const auto row            = trailing_elements(shard.source_shape);
        std::uint64_t shard_begin = 0;
        std::uint64_t held        = 0;
        for (const auto& range : shard.ranges) {
            const auto first = std::max(begin, range.begin * row);
            const auto last  = std::min(end, (range.begin + range.count) * row);
            if (first < last) {
                const auto at = shard_begin + (first - range.begin * row);
                out.parts.push_back({&parent, at, at + (last - first)});
                held += last - first;
            }
            shard_begin += range.count * row;
        }
        return held;
    }
    // A column shard narrows every row, so only whole parent rows keep a contiguous image.
    const auto columns = shard.source_shape.at(1);
    if (begin % columns || end % columns) {
        throw ArtifactError(name + ": a column shard binds only whole parent rows");
    }
    const auto width = parent.geometry.shape.at(1);
    out.parts.push_back({&parent, begin / columns * width, end / columns * width});
    return (end - begin) / columns * width;
}

} // namespace

WeightView bind_view(const ParameterReference& reference, const MaterializedArtifact& materialized,
                     int device) {
    if (reference.residency == Residency::Values) {
        throw ArtifactError(reference.name + ": owning values do not have parent backing");
    }
    WeightView out;
    out.shape             = reference.shape;
    std::uint64_t covered = 0;
    std::optional<ShardAxis> narrowed;
    std::uint64_t source_columns = 0;
    std::uint64_t shard_columns  = 0;
    for (const auto& part : reference.binding.parts) {
        const bool on_device = reference.residency == Residency::Device;
        const auto& parent   = on_device ? materialized.device_parent(part.object, device)
                                         : materialized.host_parent(part.object);
        const auto* shard = on_device ? &materialized.device_shard(part.object, device) : nullptr;
        if (!shard || !shard->sharded()) {
            if (narrowed) {
                throw ArtifactError(reference.name + ": combines sharded and complete parents");
            }
            if (part.begin >= part.end || part.end > parent.geometry.elements) {
                throw ArtifactError(reference.name + ": invalid materialized region");
            }
            covered = checked_add(covered, part.end - part.begin, reference.name);
            out.parts.push_back({&parent, part.begin, part.end});
            continue;
        }
        if (!narrowed && covered) {
            throw ArtifactError(reference.name + ": combines sharded and complete parents");
        }
        if (narrowed && *narrowed != shard->axis) {
            throw ArtifactError(reference.name + ": combines parents sharded on different axes");
        }
        narrowed = shard->axis;
        if (shard->axis == ShardAxis::Columns) {
            if ((source_columns && source_columns != shard->source_shape.at(1)) ||
                (shard_columns && shard_columns != parent.geometry.shape.at(1))) {
                throw ArtifactError(reference.name + ": column shards differ in width");
            }
            source_columns = shard->source_shape.at(1);
            shard_columns  = parent.geometry.shape.at(1);
        }
        covered = checked_add(
            covered, map_shard_part(out, parent, *shard, part.begin, part.end, reference.name),
            reference.name);
    }
    if (narrowed == ShardAxis::Rows) {
        // The device holds a leading-axis slice of the logical parameter.
        const auto row = trailing_elements(out.shape);
        if (out.shape.empty() || !covered || covered % row) {
            throw ArtifactError(reference.name + ": device shard does not hold whole rows");
        }
        out.shape.front() = covered / row;
    } else if (narrowed == ShardAxis::Columns) {
        if (out.shape.size() < 2 || out.shape.back() != source_columns) {
            throw ArtifactError(reference.name + ": column shard requires the parent row width");
        }
        out.shape.back() = shard_columns;
    }
    if (covered != weight_element_count(out.shape)) {
        throw ArtifactError(reference.name + ": materialized coverage differs from logical shape");
    }
    return out;
}

} // namespace ninfer::artifact
