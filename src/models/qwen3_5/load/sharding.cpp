#include "models/qwen3_5/load/sharding.h"

#include "artifact/reader.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::loading {
namespace {

using artifact::ShardAxis;
using artifact::SliceRange;

[[noreturn]] void reject(std::string_view name, const std::string& reason) {
    throw std::invalid_argument(std::string(name) + ": tensor parallelism " + reason);
}

LogicalShard whole(ShardAxis axis, int device = 0) { return {.axis = axis, .device = device}; }

// Splits `total` into `tp` equal contiguous blocks aligned to `units` (heads, groups or rows).
LogicalShard even(std::string_view name, ShardAxis axis, std::uint64_t total, std::uint64_t units,
                  int tp) {
    const auto parts = static_cast<std::uint64_t>(tp);
    if (!units || total % units || units % parts) {
        reject(name, "requires " + std::to_string(units) + " units of " + std::to_string(total) +
                         " to divide by " + std::to_string(tp));
    }
    LogicalShard out{.axis = axis};
    const auto count = total / parts;
    for (int device = 0; device < tp; ++device) {
        out.ranges[static_cast<std::size_t>(device)].push_back(
            {static_cast<std::uint64_t>(device) * count, count});
    }
    return out;
}

std::uint64_t leading(std::string_view name, const artifact::Shape& shape) {
    if (shape.empty()) { reject(name, "requires a shaped parameter"); }
    return shape.front();
}

std::uint64_t trailing(std::string_view name, const artifact::Shape& shape) {
    if (shape.size() != 2) { reject(name, "splits the input axis of a matrix only"); }
    return shape.back();
}

// The leaf after "text/layers/<i>/" or "mtp/layers/<i>/", or empty outside a decoder block.
std::string_view block_leaf(std::string_view name) {
    for (const std::string_view prefix : {"text/layers/", "mtp/layers/"}) {
        if (!name.starts_with(prefix)) { continue; }
        const auto slash = name.find('/', prefix.size());
        return slash == std::string_view::npos ? std::string_view{} : name.substr(slash + 1);
    }
    return {};
}

LogicalShard block_shard(std::string_view name, std::string_view leaf, const artifact::Shape& shape,
                         const TextConfig& text, int tp) {
    if (leaf == "input_norm" || leaf == "post_attention_norm" || leaf == "attention/query_norm" ||
        leaf == "attention/key_norm" || leaf == "gdn/norm") {
        return whole(ShardAxis::Replicated);
    }
    if (leaf.starts_with("attention/")) {
        const auto& a = text.attention.value();
        if (leaf == "attention/query" || leaf == "attention/gate") {
            return even(name, ShardAxis::Rows, leading(name, shape), a.num_attention_heads, tp);
        }
        if (leaf == "attention/key" || leaf == "attention/value") {
            return even(name, ShardAxis::Rows, leading(name, shape), a.num_key_value_heads, tp);
        }
        if (leaf == "attention/output") {
            return even(name, ShardAxis::Columns, trailing(name, shape), a.num_attention_heads, tp);
        }
    }
    if (leaf.starts_with("gdn/")) {
        const auto& g = text.gdn.value();
        // Value head h belongs to key head h / (value heads / key heads); contiguous per-rank
        // blocks of both keep that grouping when both head counts divide by tp.
        if (g.linear_num_key_heads % static_cast<std::uint32_t>(tp)) {
            reject(name, "requires GDN key heads to divide by " + std::to_string(tp));
        }
        const auto values = g.linear_num_value_heads;
        if (leaf == "gdn/query" || leaf == "gdn/key") {
            return even(name, ShardAxis::Rows, leading(name, shape), g.linear_num_key_heads, tp);
        }
        if (leaf == "gdn/value" || leaf == "gdn/z" || leaf == "gdn/a_projection" ||
            leaf == "gdn/b_projection" || leaf == "gdn/a_log" || leaf == "gdn/dt_bias") {
            return even(name, ShardAxis::Rows, leading(name, shape), values, tp);
        }
        if (leaf == "gdn/output") {
            return even(name, ShardAxis::Columns, trailing(name, shape), values, tp);
        }
        if (leaf == "gdn/convolution") {
            // Depthwise channels in the input projection's Q|K|V order: each rank keeps the
            // channels its own Q, K and V heads produce, in that order.
            const auto key   = g.key_width();
            const auto value = g.value_width();
            if (trailing(name, shape) != 2 * key + value) {
                reject(name, "requires Q|K|V convolution channels");
            }
            LogicalShard out{.axis = ShardAxis::Columns};
            const std::array<std::array<std::uint64_t, 3>, 3> sections{
                {{0, key, g.linear_num_key_heads},
                 {key, key, g.linear_num_key_heads},
                 {2 * key, value, values}}};
            for (const auto& [begin, width, heads] : sections) {
                const auto split = even(name, ShardAxis::Columns, width, heads, tp);
                for (std::size_t device = 0; device < split.ranges.size(); ++device) {
                    for (const auto& range : split.ranges[device]) {
                        out.ranges[device].push_back({begin + range.begin, range.count});
                    }
                }
            }
            return out;
        }
    }
    if (leaf == "mlp/gate" || leaf == "mlp/up") {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, tp);
    }
    if (leaf == "mlp/down") {
        const auto columns = trailing(name, shape);
        return even(name, ShardAxis::Columns, columns, columns, tp);
    }
    reject(name, "has no placement for this parameter");
}

bool in_component(std::string_view name, std::string_view component) {
    return name.size() > component.size() && name.starts_with(component) &&
           name[component.size()] == '/';
}

std::uint64_t row_elements(const artifact::Shape& shape) {
    return shape.size() > 1 ? weight_element_count(std::span(shape).subspan(1)) : 1;
}

// Sorts, rejects overlaps and joins adjacent ranges.
void normalize(std::vector<SliceRange>& ranges, const std::string& id) {
    std::sort(ranges.begin(), ranges.end(),
              [](const SliceRange& a, const SliceRange& b) { return a.begin < b.begin; });
    std::vector<SliceRange> out;
    for (const auto& range : ranges) {
        if (!out.empty() && out.back().begin + out.back().count > range.begin) {
            throw std::invalid_argument(id + ": tensor-parallel ranges overlap");
        }
        if (!out.empty() && out.back().begin + out.back().count == range.begin) {
            out.back().count += range.count;
        } else {
            out.push_back(range);
        }
    }
    ranges = std::move(out);
}

// Every index of [0, extent) belongs to exactly one device.
void require_partition(const artifact::ShardPlacement& placement, int tp, std::uint64_t extent,
                       const std::string& id) {
    std::vector<SliceRange> all;
    for (int device = 0; device < tp; ++device) {
        const auto& ranges = placement.device_ranges[static_cast<std::size_t>(device)];
        if (ranges.empty()) {
            throw std::invalid_argument(id + ": tensor-parallel placement leaves device " +
                                        std::to_string(device) + " empty");
        }
        all.insert(all.end(), ranges.begin(), ranges.end());
    }
    normalize(all, id);
    if (all.size() != 1 || all.front().begin != 0 || all.front().count != extent) {
        throw std::invalid_argument(id + ": tensor-parallel ranges do not cover the parent");
    }
}

struct ParentState {
    std::optional<artifact::ShardPlacement> placement;
    std::string first; // First logical parameter, for diagnostics.
};

void merge_axis(ParentState& state, const LogicalShard& shard, const std::string& name,
                const std::string& id) {
    if (!state.placement) {
        state.placement = artifact::ShardPlacement{.axis = shard.axis, .device = shard.device};
        state.first     = name;
        return;
    }
    if (state.placement->axis != shard.axis || state.placement->device != shard.device) {
        throw std::invalid_argument(id + ": " + state.first + " and " + name +
                                    " share a parent with different tensor-parallel placements");
    }
}

} // namespace

LogicalShard logical_shard(std::string_view name, const artifact::Shape& shape,
                           const Config& config, const LoadOptions& options) {
    const int tp = options.tp;
    if (tp < 1 || tp > static_cast<int>(artifact::kMaximumDevices)) {
        throw std::invalid_argument("tensor parallelism must be 1 or 2");
    }
    if (tp == 1) { return {}; }
    if (options.vision_rank < 0 || options.vision_rank >= tp) {
        throw std::invalid_argument("vision_rank must name a tensor-parallel rank");
    }
    if (in_component(name, "vision")) {
        return whole(ShardAxis::SingleDevice, options.vision_rank);
    }
    if (in_component(name, "dflash") || in_component(name, "dflash2") ||
        in_component(name, "proposal")) {
        return whole(ShardAxis::PrimaryOnly);
    }
    if (name == "text/token_embedding" || name == "text/final_norm" ||
        name == "mtp/embedding_norm" || name == "mtp/hidden_norm" || name == "mtp/final_norm") {
        return whole(ShardAxis::Replicated);
    }
    if (name == "text/output_head") {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, tp);
    }
    if (name == "mtp/input_projection") {
        // [hidden, 2 * hidden] contracts the packed [embedding; hidden] input.
        const auto columns = trailing(name, shape);
        return even(name, ShardAxis::Columns, columns, columns, tp);
    }
    if (const auto leaf = block_leaf(name); !leaf.empty()) {
        if (!std::holds_alternative<DenseConfig>(config.text.ffn)) {
            reject(name, "supports the dense architecture only");
        }
        return block_shard(name, leaf, shape, config.text, tp);
    }
    reject(name, "has no placement for this parameter");
}

std::vector<std::optional<artifact::ShardPlacement>>
parent_placements(std::span<const PendingWeight> weights, std::size_t object_count,
                  const GeometryLookup& geometry, const Config& config,
                  const LoadOptions& options) {
    std::vector<ParentState> states(object_count);
    std::vector<std::string> ids(object_count);
    for (const auto& weight : weights) {
        const auto& reference = weight.reference;
        const auto& name      = reference.name;
        const auto shard      = logical_shard(name, reference.shape, config, options);
        const auto row        = row_elements(reference.shape);
        std::uint64_t logical = 0; // Logical element offset of the current part.
        for (std::size_t i = 0; i < reference.binding.parts.size(); ++i) {
            const auto& part   = reference.binding.parts[i];
            const auto& parent = geometry(part.object);
            auto& state        = states.at(part.object.index);
            auto& id           = ids[part.object.index];
            if (id.empty()) {
                id = i < weight.source_objects.size() ? weight.source_objects[i] : name;
            }
            merge_axis(state, shard, name, id);
            if (shard.axis == ShardAxis::Rows) {
                // Logical row ranges intersect this part's element range as whole parent rows.
                const auto parent_row = row_elements(parent.shape);
                const auto length     = part.end - part.begin;
                for (std::size_t device = 0; device < shard.ranges.size(); ++device) {
                    for (const auto& range : shard.ranges[device]) {
                        const auto first = std::max(range.begin * row, logical);
                        const auto last =
                            std::min((range.begin + range.count) * row, logical + length);
                        if (first >= last) { continue; }
                        const auto begin = part.begin + (first - logical);
                        const auto end   = part.begin + (last - logical);
                        if (begin % parent_row || end % parent_row) {
                            throw std::invalid_argument(
                                name + ": a row split does not fall on rows of parent " + id);
                        }
                        state.placement->device_ranges[device].push_back(
                            {begin / parent_row, (end - begin) / parent_row});
                    }
                }
            } else if (shard.axis == ShardAxis::Columns) {
                // A column split narrows every parent row: the parent must have the logical row
                // width and this parameter must bind whole parent rows.
                const auto columns = reference.shape.back();
                if (parent.shape.size() != 2 || parent.shape[1] != columns ||
                    part.begin % columns || part.end % columns) {
                    throw std::invalid_argument(name + ": a column split requires whole rows of " +
                                                id);
                }
                auto& ranges = state.placement->device_ranges;
                if (std::all_of(ranges.begin(), ranges.end(),
                                [](const auto& list) { return list.empty(); })) {
                    ranges = shard.ranges;
                } else if (ranges != shard.ranges) {
                    throw std::invalid_argument(id + ": " + state.first + " and " + name +
                                                " split the columns of one parent differently");
                }
            }
            logical += part.end - part.begin;
        }
    }
    std::vector<std::optional<artifact::ShardPlacement>> out(object_count);
    for (std::size_t object = 0; object < object_count; ++object) {
        auto& placement = states[object].placement;
        if (!placement) { continue; }
        if (artifact::is_sharded(placement->axis)) {
            const auto& parent = geometry(artifact::ObjectHandle{object});
            for (auto& ranges : placement->device_ranges) { normalize(ranges, ids[object]); }
            const auto extent = placement->axis == ShardAxis::Rows
                                    ? parent.elements / row_elements(parent.shape)
                                    : parent.shape.at(1);
            require_partition(*placement, options.tp, extent, ids[object]);
        }
        out[object] = std::move(placement);
    }
    return out;
}

void install_shard_resolver(artifact::Binder& binder, std::span<const PendingWeight> weights,
                            const Config& config, const LoadOptions& options) {
    if (options.tp == 1) { return; }
    if (binder.device_count() != options.tp) {
        throw std::logic_error("Binder device count differs from tensor parallelism");
    }
    const auto& reader = binder.reader();
    auto placements    = parent_placements(
        weights, reader.directory().objects.size(),
        [&](artifact::ObjectHandle object) -> const WeightGeometry& {
            return reader.geometry(object);
        },
        config, options);
    binder.set_shard_resolver([placements = std::move(placements),
                               &reader](artifact::ObjectHandle object, const WeightGeometry&) {
        const auto& placement = placements.at(object.index);
        if (!placement) {
            throw artifact::ArtifactError(
                artifact::object_id(reader.directory().object(object)) +
                ": device parent has no logical tensor-parallel placement");
        }
        return *placement;
    });
}

} // namespace ninfer::models::qwen3_5::loading
