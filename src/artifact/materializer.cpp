#include "artifact/materializer.h"

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "core/startup.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024 * 1024;
constexpr std::size_t kMaximumSlotCount = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string(operation) + ": " + cudaGetErrorName(status) + ": " +
                            cudaGetErrorString(status));
    }
}

// Selects each upload device in turn and restores the caller's device. A single-device upload
// leaves the current device alone, as it always has.
class DeviceSelection {
public:
    explicit DeviceSelection(std::span<DeviceContext* const> devices) : devices_(devices) {
        if (devices_.size() > 1) {
            check_cuda(cudaGetDevice(&caller_), "query current device");
            active_ = true;
        }
    }

    ~DeviceSelection() {
        if (active_) { (void)cudaSetDevice(caller_); }
    }

    DeviceSelection(const DeviceSelection&)            = delete;
    DeviceSelection& operator=(const DeviceSelection&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return devices_.size(); }

    void select(std::size_t index) const {
        if (active_) {
            check_cuda(cudaSetDevice(devices_[index]->device), "select weight upload device");
        }
    }

private:
    std::span<DeviceContext* const> devices_;
    int caller_  = 0;
    bool active_ = false;
};

// A slot is refilled only after every device that read it has finished its copies. Each
// device's completion event is created on that device.
class Slot {
public:
    Slot(std::size_t bytes, const DeviceSelection& selection) : buffer(bytes) {
        try {
            for (std::size_t i = 0; i < selection.size(); ++i) {
                selection.select(i);
                check_cuda(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming),
                           "create weight staging completion event");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~Slot() { release(); }

    Slot(const Slot&)            = delete;
    Slot& operator=(const Slot&) = delete;

    void wait() {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (pending[i]) {
                check_cuda(cudaEventSynchronize(events[i]), "wait for weight staging transfer");
                pending[i] = false;
            }
        }
    }

    PinnedHostBuffer buffer;
    std::array<cudaEvent_t, kMaximumDevices> events{};
    std::array<bool, kMaximumDevices> pending{};

private:
    void release() noexcept {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (pending[i]) { (void)cudaEventSynchronize(events[i]); }
            if (events[i]) { (void)cudaEventDestroy(events[i]); }
            events[i]  = nullptr;
            pending[i] = false;
        }
    }
};

// Also covers failure between a queued copy and its event record. Destruct before slots.
struct TransferCompletion {
    std::span<DeviceContext* const> devices;
    bool pending = true;

    ~TransferCompletion() {
        if (pending) {
            for (auto* device : devices) { (void)cudaStreamSynchronize(device->transfer_stream); }
        }
    }

    void finish() {
        for (auto* device : devices) {
            check_cuda(cudaStreamSynchronize(device->transfer_stream), "complete weight upload");
        }
        pending = false;
    }
};

struct CopyRange {
    std::size_t file       = 0;
    std::uint64_t begin    = 0;
    std::uint64_t end      = 0;
    std::byte* destination = nullptr;
    std::size_t device     = 0;
};

struct ReadSpan {
    std::size_t file    = 0;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

float read_divisor(const Reader& reader, ObjectHandle handle, const WeightGeometry& geometry,
                   std::span<const std::byte> host, MaterializationStats& stats) {
    if (geometry.format != QType::NVFP4) { return 0.0F; }
    std::array<std::byte, 4> word{};
    if (!host.empty()) {
        std::copy_n(host.data() + geometry.divisor_offset, word.size(), word.data());
    } else {
        const auto& object = reader.directory().tensor(handle);
        reader.read_into(checked_add(object.offset, geometry.divisor_offset, "weight divisor"),
                         word);
        stats.read_bytes = checked_add(stats.read_bytes, word.size(), "read bytes");
    }
    const auto value = std::bit_cast<float>(read_u32_le(word.data()));
    if (!std::isfinite(value) || value <= 0) {
        throw ArtifactError(reader.directory().tensor(handle).id +
                            ": invalid NVFP4 weight divisor");
    }
    return value;
}

} // namespace

const WeightParent& MaterializedArtifact::device_parent(ObjectHandle handle, int device) const {
    if (!has_device(handle, device)) { throw ArtifactError("object has no device weight backing"); }
    return *objects_[handle.index].device[static_cast<std::size_t>(device)].parent;
}

const DeviceShard& MaterializedArtifact::device_shard(ObjectHandle handle, int device) const {
    if (!has_device(handle, device)) { throw ArtifactError("object has no device weight backing"); }
    return objects_[handle.index].device[static_cast<std::size_t>(device)].shard;
}

const WeightParent& MaterializedArtifact::host_parent(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].host) {
        throw ArtifactError("object has no Host weight backing");
    }
    return *objects_[handle.index].host;
}

std::span<const std::byte> MaterializedArtifact::host_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].host_data.empty()) {
        throw ArtifactError("object has no retained Host bytes");
    }
    return objects_[handle.index].host_data;
}

bool MaterializedArtifact::has_device(ObjectHandle handle, int device) const noexcept {
    return device >= 0 && device < stats_.device_count && handle.index < objects_.size() &&
           objects_[handle.index].device[static_cast<std::size_t>(device)].parent.has_value();
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 std::span<DeviceContext* const> devices,
                                 const StartupObserver* startup_observer) {
    if (plan.source != &reader || plan.object_count != reader.directory().objects.size()) {
        throw ArtifactError("materialization plan belongs to another load session");
    }
    if (plan.device_count < 1 || plan.device_count > static_cast<int>(kMaximumDevices) ||
        devices.size() != static_cast<std::size_t>(plan.device_count)) {
        throw ArtifactError("materialization devices differ from the plan device count");
    }
    const StartupObserver no_observer;
    const auto& observer = startup_observer ? *startup_observer : no_observer;
    std::uint64_t total  = 0;
    for (const auto& placement : plan.device_objects) {
        if (placement.copies.empty()) {
            total = checked_add(total, placement.bytes, "device payload bytes");
        }
        for (const auto& copy : placement.copies) {
            total = checked_add(total, copy.bytes, "device payload bytes");
        }
    }
    StartupPhaseScope phase(observer, StartupPhase::WeightsMaterialize, StartupProgressUnit::Bytes,
                            total);
    DeviceSelection selection(devices);
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    out.stats_.device_count          = plan.device_count;
    out.stats_.file_bytes            = reader.file_bytes();
    out.stats_.read_bytes            = plan.prior_read_bytes;
    out.stats_.owned_value_bytes     = plan.owned_value_bytes;
    out.stats_.device_capacity_bytes = plan.device_capacity_bytes;
    out.stats_.host_object_count     = plan.host_objects.size();
    // A shard's plane alignment gaps are bytes no copy writes; they read as zero, like the
    // padding of a stored parent, so a device holding a shard gets a ZeroFill::Yes arena, whose
    // zeroing completes before the constructor returns, exactly as every arena did before the
    // opt-in zero fill. A stream-ordered memset on the transfer stream was tried instead; the
    // tp 2 MTP Engine failure that appeared at the same time (cudaGraphExecUpdate rejecting a
    // memcpy node with cudaGraphExecUpdateErrorParametersChanged) later turned out to be
    // intermittent and unrelated, and is now handled by re-instantiating the rejected profile
    // (see DecodeGraphExecutable::update_or_reinstantiate). The constructor path is kept because
    // it is the simplest way to guarantee the zeroed padding before any shard upload starts.
    std::array<bool, kMaximumDevices> holds_shard{};
    for (const auto& placement : plan.device_objects) {
        if (is_sharded(placement.axis) && placement.device >= 0 &&
            placement.device < plan.device_count) {
            holds_shard[static_cast<std::size_t>(placement.device)] = true;
        }
    }
    std::uint64_t capacity = 0;
    for (std::size_t device = 0; device < devices.size(); ++device) {
        const auto bytes = plan.per_device_capacity_bytes[device];
        capacity         = checked_add(capacity, bytes, "device capacity");
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError("device backing exceeds size_t");
        }
        out.stats_.per_device_capacity_bytes[device] = bytes;
        if (bytes) {
            selection.select(device);
            try {
                out.arenas_[device] = std::make_unique<DeviceArena>(
                    static_cast<std::size_t>(bytes),
                    holds_shard[device] ? ZeroFill::Yes : ZeroFill::No);
            } catch (const std::exception& error) {
                throw ArtifactError("device " + std::to_string(devices[device]->device) +
                                    " weight arena of " + std::to_string(bytes) +
                                    " bytes: " + error.what());
            }
        }
    }
    if (capacity != plan.device_capacity_bytes) {
        throw ArtifactError("device capacity differs from its per-device sum");
    }
    for (auto& placement : plan.host_objects) {
        reader.validate_object(placement.object);
        auto& storage = out.objects_.at(placement.object.index);
        if (!storage.host_data.empty()) { throw ArtifactError("duplicate Host placement"); }
        const auto& object = reader.directory().object(placement.object);
        if (placement.data.empty()) {
            placement.data = reader.read_object(placement.object);
            out.stats_.read_bytes =
                checked_add(out.stats_.read_bytes, placement.data.size(), "Host read bytes");
        }
        if (placement.data.size() != object_bytes(object)) {
            throw ArtifactError("Host placement size differs from object");
        }
        storage.host_data = std::move(placement.data);
        out.stats_.retained_host_bytes =
            checked_add(out.stats_.retained_host_bytes, storage.host_data.size(), "retained bytes");
        if (std::holds_alternative<TensorObject>(object)) {
            const auto& geometry = reader.geometry(placement.object);
            const auto divisor =
                read_divisor(reader, placement.object, geometry, storage.host_data, out.stats_);
            storage.host = WeightParent{geometry, storage.host_data.data(), divisor};
        }
    }
    std::vector<CopyRange> ranges;
    const auto add_ranges = [&](const TensorObject& descriptor, std::uint64_t source,
                                std::uint64_t bytes, std::byte* destination, std::size_t device) {
        for (const auto& segment :
             reader.segments(checked_add(descriptor.offset, source, "copy source"), bytes)) {
            ranges.push_back({segment.file_index, segment.file_offset,
                              checked_add(segment.file_offset, segment.bytes, "copy range"),
                              destination + segment.destination_offset, device});
        }
    };
    for (const auto& placement : plan.device_objects) {
        if (placement.device < 0 || placement.device >= plan.device_count) {
            throw ArtifactError("device placement names a device outside the plan");
        }
        const auto device    = static_cast<std::size_t>(placement.device);
        const auto& geometry = reader.geometry(placement.object);
        auto& object         = out.objects_.at(placement.object.index);
        auto& backing        = object.device[device];
        auto& arena          = out.arenas_[device];
        const bool sharded   = is_sharded(placement.axis);
        std::optional<TensorSlice> slice;
        if (sharded) { slice = tensor_slice(geometry, placement.axis, placement.ranges); }
        const auto& resident = slice ? slice->geometry : geometry;
        if (backing.parent || !arena || resident.bytes != placement.bytes ||
            (slice ? slice->copies != placement.copies
                   : !placement.ranges.empty() || !placement.copies.empty())) {
            throw ArtifactError("invalid or duplicate device placement");
        }
        auto storage      = arena->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                               static_cast<std::size_t>(placement.alignment));
        const auto offset = static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                                       static_cast<std::byte*>(arena->base()));
        if (offset != placement.offset) {
            throw ArtifactError("device offset differs from materialization plan");
        }
        // One divisor read per parent, shared by every device that holds it.
        std::optional<float> divisor;
        if (object.host) { divisor = object.host->weight_scale_divisor; }
        for (const auto& other : object.device) {
            if (!divisor && other.parent) { divisor = other.parent->weight_scale_divisor; }
        }
        if (!divisor) {
            divisor = read_divisor(reader, placement.object, geometry, {}, out.stats_);
        }
        backing.parent =
            WeightParent{resident, static_cast<const std::byte*>(storage.data), *divisor};
        backing.shard          = {placement.axis, placement.ranges, geometry.shape};
        auto* destination      = static_cast<std::byte*>(storage.data);
        const auto& descriptor = reader.directory().tensor(placement.object);
        if (sharded) {
            for (const auto& copy : placement.copies) {
                add_ranges(descriptor, copy.source_offset, copy.bytes,
                           destination + copy.dest_offset, device);
            }
            out.stats_.sharded_bytes[device] += placement.bytes;
        } else {
            add_ranges(descriptor, 0, descriptor.bytes, destination, device);
            auto& kind = placement.axis == ShardAxis::Replicated ? out.stats_.replicated_bytes
                                                                 : out.stats_.local_bytes;
            kind[device] += placement.bytes;
        }
    }
    for (const auto& object : out.objects_) {
        if (std::any_of(object.device.begin(), object.device.end(),
                        [](const auto& backing) { return backing.parent.has_value(); })) {
            ++out.stats_.device_object_count;
        }
    }
    if (ranges.empty()) {
        phase.complete();
        return out;
    }
    // Every device byte is written once. Sources may repeat: each device reads its own copy of
    // a replicated parent, and all copies are cut from the staged chunk independently below.
    {
        std::vector<const CopyRange*> written;
        written.reserve(ranges.size());
        for (const auto& range : ranges) { written.push_back(&range); }
        std::sort(written.begin(), written.end(), [](const auto* a, const auto* b) {
            return std::tie(a->device, a->destination) < std::tie(b->device, b->destination);
        });
        for (std::size_t i = 1; i < written.size(); ++i) {
            const auto& previous = *written[i - 1];
            if (previous.device == written[i]->device &&
                previous.destination + (previous.end - previous.begin) > written[i]->destination) {
                throw ArtifactError("device destination ranges overlap");
            }
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return std::tie(a.file, a.begin) < std::tie(b.file, b.begin);
    });
    std::vector<ReadSpan> spans;
    std::uint64_t aligned_bytes = 0;
    for (const auto& range : ranges) {
        const auto begin = range.begin / kPayloadAlignment * kPayloadAlignment;
        if (spans.empty() || spans.back().file != range.file ||
            begin > align_up(spans.back().end, kPayloadAlignment, "direct range")) {
            spans.push_back({range.file, begin, range.end});
        } else {
            spans.back().end = std::max(spans.back().end, range.end);
        }
    }
    for (const auto& span : spans) {
        aligned_bytes = checked_add(
            aligned_bytes, align_up(span.end - span.begin, kPayloadAlignment, "direct range bytes"),
            "direct bytes");
    }
    const auto slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_bytes));
    const auto slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    out.stats_.peak_staging_bytes = slot_bytes * slot_count;
    StartupPhaseScope pin_phase(observer, StartupPhase::WeightsStagingPin,
                                StartupProgressUnit::Bytes, out.stats_.peak_staging_bytes);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes, selection));
    }
    pin_phase.complete();
    TransferCompletion completion{devices};
    std::size_t next_slot = 0;
    // Ranges before `live` ended before the current chunk. Later ranges may start inside a
    // range that is still being copied, so each chunk scans from `live` to its end.
    std::size_t live     = 0;
    std::uint64_t copied = 0;
    const auto start     = std::chrono::steady_clock::now();
    for (const auto& span : spans) {
        for (auto source = span.begin; source < span.end; source += slot_bytes) {
            auto& slot = *slots[next_slot++ % slot_count];
            slot.wait();
            const auto remaining = span.end - source;
            const auto request   = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes, align_up(remaining, kPayloadAlignment, "direct block bytes")));
            const auto received  = reader.read_direct(
                span.file, source, {static_cast<std::byte*>(slot.buffer.data()), request});
            if (received < std::min<std::uint64_t>(request, remaining)) {
                throw ArtifactError("direct read ended before the required payload");
            }
            out.stats_.read_bytes = checked_add(out.stats_.read_bytes, received, "read bytes");
            const auto chunk_end  = checked_add(source, received, "read block end");
            while (live < ranges.size() &&
                   (ranges[live].file < span.file ||
                    (ranges[live].file == span.file && ranges[live].end <= source))) {
                ++live;
            }
            for (std::size_t device = 0; device < devices.size(); ++device) {
                bool fed = false;
                for (auto i = live; i < ranges.size() && ranges[i].file == span.file &&
                                    ranges[i].begin < chunk_end;
                     ++i) {
                    const auto& range = ranges[i];
                    const auto begin  = std::max(source, range.begin);
                    const auto end    = std::min(chunk_end, range.end);
                    if (range.device != device || begin >= end) { continue; }
                    if (!fed) { selection.select(device); }
                    check_cuda(cudaMemcpyAsync(range.destination + (begin - range.begin),
                                               static_cast<const std::byte*>(slot.buffer.data()) +
                                                   (begin - source),
                                               static_cast<std::size_t>(end - begin),
                                               cudaMemcpyHostToDevice,
                                               devices[device]->transfer_stream),
                               "upload weight bytes");
                    copied = checked_add(copied, end - begin, "copied bytes");
                    out.stats_.per_device_h2d_bytes[device] += end - begin;
                    fed = true;
                }
                if (fed) {
                    check_cuda(
                        cudaEventRecord(slot.events[device], devices[device]->transfer_stream),
                        "record weight staging completion");
                    slot.pending[device] = true;
                }
            }
            phase.progress(copied, total);
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    completion.finish();
    if (copied != total) { throw ArtifactError("incomplete device upload"); }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    slots.clear();
    phase.complete(copied, total);
    return out;
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 DeviceContext& device, const StartupObserver* startup_observer) {
    if (plan.device_count != 1) {
        throw ArtifactError("a multi-device materialization plan requires an ExecutionContext");
    }
    DeviceContext* const devices[] = {&device};
    return materialize(reader, std::move(plan), devices, startup_observer);
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 ExecutionContext& execution,
                                 const StartupObserver* startup_observer) {
    if (plan.device_count != execution.tp) {
        throw ArtifactError("materialization plan device count differs from the tensor-parallel "
                            "degree");
    }
    std::array<DeviceContext*, kMaximumDevices> devices{};
    for (int i = 0; i < execution.tp; ++i) {
        devices[static_cast<std::size_t>(i)] = &*execution.dev[static_cast<std::size_t>(i)];
    }
    return materialize(
        reader, std::move(plan),
        std::span<DeviceContext* const>(devices.data(), static_cast<std::size_t>(execution.tp)),
        startup_observer);
}

} // namespace ninfer::artifact
