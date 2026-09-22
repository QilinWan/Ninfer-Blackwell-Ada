#pragma once

#include "artifact/schema.h"
#include "artifact/slices.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

// One device's backing of one parent. A complete parent has no ranges or copies and receives
// its encoded bytes verbatim; a Rows/Columns shard is the parent slice named by `ranges`, filled
// by `copies` (tensor_slice) and described by the shard geometry.
struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
    int device              = 0;
    ShardAxis axis          = ShardAxis::Replicated;
    std::vector<SliceRange> ranges;
    std::vector<PlaneCopy> copies;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

struct MaterializationPlan {
    const Reader* source                = nullptr;
    std::size_t object_count            = 0;
    int device_count                    = 1;
    std::uint64_t device_capacity_bytes = 0; // Sum over devices.
    std::array<std::uint64_t, kMaximumDevices> per_device_capacity_bytes{};
    std::uint64_t prior_read_bytes  = 0;
    std::uint64_t owned_value_bytes = 0;
    // Object order, then device order within one object.
    std::vector<DevicePlacement> device_objects;
    std::vector<HostPlacement> host_objects;
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0; // Sum over devices.
    std::uint64_t device_capacity_bytes = 0; // Sum over devices.
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    std::size_t device_object_count     = 0; // Distinct parents with any device backing.
    std::size_t host_object_count       = 0;
    double upload_seconds               = 0;
    int device_count                    = 1;
    std::array<std::uint64_t, kMaximumDevices> per_device_h2d_bytes{};
    std::array<std::uint64_t, kMaximumDevices> per_device_capacity_bytes{};
    // Placed bytes by kind, without alignment gaps: this device's slice of a Rows/Columns parent,
    // a complete Replicated parent, or a complete parent held by this device only.
    std::array<std::uint64_t, kMaximumDevices> sharded_bytes{};
    std::array<std::uint64_t, kMaximumDevices> replicated_bytes{};
    std::array<std::uint64_t, kMaximumDevices> local_bytes{};
};

// How a device parent relates to its artifact parent. `ranges` is empty for a complete parent;
// otherwise the device parent concatenates those `axis` ranges of a parent of `source_shape`.
struct DeviceShard {
    ShardAxis axis = ShardAxis::Replicated;
    std::vector<SliceRange> ranges;
    Shape source_shape;

    [[nodiscard]] bool sharded() const noexcept { return !ranges.empty(); }
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle, int device = 0) const;
    [[nodiscard]] const DeviceShard& device_shard(ObjectHandle handle, int device = 0) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle, int device = 0) const noexcept;

    [[nodiscard]] int device_count() const noexcept { return stats_.device_count; }

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&,
                                            std::span<DeviceContext* const>,
                                            const StartupObserver*);

    struct DeviceStorage {
        std::optional<WeightParent> parent;
        DeviceShard shard;
    };

    struct ObjectStorage {
        std::array<DeviceStorage, kMaximumDevices> device;
        std::optional<WeightParent> host;
        std::vector<std::byte> host_data;
    };

    std::array<std::unique_ptr<DeviceArena>, kMaximumDevices> arenas_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

// Uploads the plan to devices[0, plan.device_count), device i receiving the placements for i.
[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               std::span<DeviceContext* const> devices,
                                               const StartupObserver* startup_observer = nullptr);

// Requires plan.device_count == 1.
[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               DeviceContext& device,
                                               const StartupObserver* startup_observer = nullptr);

// Requires plan.device_count == execution.tp; placements for device i use execution.dev[i].
[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               ExecutionContext& execution,
                                               const StartupObserver* startup_observer = nullptr);

} // namespace ninfer::artifact
