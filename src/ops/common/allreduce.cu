// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition only: the transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// Both collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull peer source into own storage;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); local combine (allreduce_sum only)
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "ninfer/ops/peer_mailbox.h"

#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_two_devices(const ExecutionContext& ec, const char* message) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), message);
    require(ec.dev[0]->device != ec.dev[1]->device, message);
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04),
// which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

// CAPTURED MAILBOX SELECTION. Returns the mailbox attached to `events` when every predicate holds:
//
//   * the mailbox serves this ExecutionContext's device pair;
//   * the payload is whole 16-byte vectors within one slot, at 16-byte aligned addresses (the
//     exchange kernel's vector contract);
//   * BOTH ranks' streams are capturing, into the SAME capture. The exchange has no host reset
//     point and no event ordering: its two kernels must be nodes of one graph that every launch
//     runs on both devices. A rank whose stream is not in that capture would run its half
//     eagerly, spin without a partner and report a hang.
//
// Every predicate the mailbox fails is a collective that runs the staged path, unchanged. Leaves
// the current device changed; the caller holds a CurrentDeviceGuard.
PeerMailbox* captured_mailbox(const std::array<Tensor, 2>& buffer, std::size_t bytes,
                              const ExecutionContext& ec, const PeerEvents& events) {
    for (int rank = 0; rank < 2; ++rank) {
        if ((reinterpret_cast<std::uintptr_t>(buffer[rank].data) % 16) != 0) { return nullptr; }
    }
    return detail::captured_mailbox_for(bytes, ec, events);
}

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    if (pair[0] == pair[1]) { return false; }

    int forward = 0;
    int reverse = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, pair[0], pair[1]));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, pair[1], pair[0]));
    // Asymmetric support is not a usable transport for a symmetric collective: fall back to the
    // staged path rather than enabling one direction only.
    if (forward == 0 || reverse == 0) { return false; }

    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(pair[rank]);
        const cudaError_t status = cudaDeviceEnablePeerAccess(pair[1 - rank], 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) {
            // Already enabled by an earlier call; clear the sticky runtime error so the next
            // cudaGetLastError() in an unrelated launcher does not observe it.
            cudaGetLastError();
            continue;
        }
        CUDA_CHECK(status);
    }
    return true;
}

PeerEvents::PeerEvents(const ExecutionContext& ec) {
    require_two_devices(ec, "PeerEvents: requires an ExecutionContext with two distinct devices");
    const CurrentDeviceGuard guard;
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    cudaEvent_t created[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int slot = 0; slot < 4; ++slot) {
        const int rank             = slot % 2;
        const cudaError_t creation = cudaSetDevice(ec.dev[rank]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    inputs_ready_ = {created[0], created[1]};
    pull_done_    = {created[2], created[3]};
}

PeerEvents::~PeerEvents() {
    for (std::array<cudaEvent_t, 2>* group : {&inputs_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : inputs_ready_(other.inputs_ready_), pull_done_(other.pull_done_), mailbox_(other.mailbox_),
      staged_scopes_(other.staged_scopes_), split_lanes_(std::move(other.split_lanes_)) {
    other.inputs_ready_  = {nullptr, nullptr};
    other.pull_done_     = {nullptr, nullptr};
    other.mailbox_       = nullptr;
    other.staged_scopes_ = 0;
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    inputs_ready_.swap(other.inputs_ready_);
    pull_done_.swap(other.pull_done_);
    std::swap(mailbox_, other.mailbox_);
    std::swap(staged_scopes_, other.staged_scopes_);
    split_lanes_.swap(other.split_lanes_);
    return *this;
}

PeerMailbox* detail::captured_mailbox_for(std::size_t bytes, const ExecutionContext& ec,
                                          const PeerEvents& events) {
    PeerMailbox* mailbox = events.mailbox();
    if (mailbox == nullptr || events.staged_only() || !mailbox->serves(ec)) { return nullptr; }
    if ((bytes % 16) != 0 || bytes > mailbox->slot_bytes()) { return nullptr; }
    unsigned long long capture[2] = {0, 0};
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamGetCaptureInfo(local.stream, &status, &capture[rank]));
        if (status != cudaStreamCaptureStatusActive) { return nullptr; }
    }
    return capture[0] == capture[1] ? mailbox : nullptr;
}

namespace {
// Chunk of a lane's transfer: the peer's host-to-device copy of chunk c overlaps the
// device-to-host copy of chunk c+1 (the two directions run on the two boards' links).
constexpr std::size_t kLaneChunkBytes = std::size_t{2} << 20;
} // namespace

SplitLanes::SplitLanes(const ExecutionContext& ec) {
    require_two_devices(ec, "SplitLanes: requires an ExecutionContext with two distinct devices");
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        devices_[r]  = ec.dev[rank]->device;
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&send_[r], cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&receive_[r], cudaStreamNonBlocking));
        for (Lane& lane : lanes_) {
            CUDA_CHECK(cudaEventCreateWithFlags(&lane.begun[r], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&lane.received[r], cudaEventDisableTiming));
        }
    }
}

SplitLanes::~SplitLanes() {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) { previous = -1; }
    for (std::size_t rank = 0; rank < 2; ++rank) {
        if (devices_[rank] < 0 || cudaSetDevice(devices_[rank]) != cudaSuccess) { continue; }
        for (cudaStream_t stream : {send_[rank], receive_[rank]}) {
            if (stream != nullptr) {
                (void)cudaStreamSynchronize(stream);
                (void)cudaStreamDestroy(stream);
            }
        }
        for (Lane& lane : lanes_) {
            for (cudaEvent_t event : {lane.begun[rank], lane.received[rank]}) {
                if (event != nullptr) { (void)cudaEventDestroy(event); }
            }
            for (cudaEvent_t event : lane.chunk_done[rank]) {
                if (event != nullptr) { (void)cudaEventDestroy(event); }
            }
        }
    }
    // The peers' host-to-device reads ran on the ranks' own streams: drain both devices before
    // the buffers go.
    for (std::size_t rank = 0; rank < 2; ++rank) {
        if (devices_[rank] >= 0 && cudaSetDevice(devices_[rank]) == cudaSuccess) {
            (void)cudaDeviceSynchronize();
        }
    }
    for (Lane& lane : lanes_) {
        for (std::byte* host : lane.host) {
            if (host != nullptr) { (void)cudaFreeHost(host); }
        }
    }
    if (previous >= 0) { (void)cudaSetDevice(previous); }
}

bool SplitLanes::serves(const ExecutionContext& ec) const noexcept {
    return ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value() &&
           ec.dev[0]->device == devices_[0] && ec.dev[1]->device == devices_[1];
}

void SplitLanes::reserve(Lane& lane, std::size_t bytes, std::size_t chunks,
                         const ExecutionContext& ec) {
    if (bytes > lane.capacity) {
        // An earlier call of this lane may still read or write the old buffers: drain both.
        for (int rank = 0; rank < 2; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        for (std::byte*& host : lane.host) {
            if (host != nullptr) {
                CUDA_CHECK(cudaFreeHost(host));
                host = nullptr;
            }
        }
        lane.capacity = 0;
        for (std::byte*& host : lane.host) {
            void* p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
            host = static_cast<std::byte*>(p);
        }
        lane.capacity = bytes;
    }
    for (int rank = 0; rank < 2; ++rank) {
        auto& events = lane.chunk_done[static_cast<std::size_t>(rank)];
        if (events.size() >= chunks) { continue; }
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        while (events.size() < chunks) {
            cudaEvent_t event = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            events.push_back(event);
        }
    }
}

void SplitLanes::sum_begin(int lane_index, const std::array<Tensor, 2>& buffer,
                           const std::array<Tensor, 2>& staging, const ExecutionContext& ec) {
    require(serves(ec), "SplitLanes: serves another device pair");
    require(lane_index >= 0 && lane_index < kCount, "SplitLanes: sum lane out of range");
    Lane& lane = lanes_[static_cast<std::size_t>(lane_index)];
    require(!lane.active, "SplitLanes: sum_begin on an open lane");
    const std::size_t bytes = buffer[0].bytes();
    require(bytes > 0 && buffer[1].bytes() == bytes && staging[0].bytes() == bytes &&
                staging[1].bytes() == bytes,
            "SplitLanes: sum operands must share one size");
    const CurrentDeviceGuard guard;
    const std::size_t chunks = (bytes + kLaneChunkBytes - 1) / kLaneChunkBytes;
    reserve(lane, bytes, chunks, ec);
    // Both operands are complete here on the ranks' streams; staging[r] and the lane's host
    // buffers are free, since the lane's previous sum_finish (earlier on the same streams)
    // waited for every copy of its call.
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(lane.begun[r], ec.dev[rank]->stream));
        CUDA_CHECK(cudaStreamWaitEvent(send_[r], lane.begun[r], 0));
        CUDA_CHECK(cudaStreamWaitEvent(receive_[r], lane.begun[r], 0));
    }
    // Leg 1, both ranks: operand -> the lane's host buffer, one event per chunk.
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const auto* source = static_cast<const std::byte*>(buffer[r].data);
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t offset = c * kLaneChunkBytes;
            const std::size_t n      = std::min(kLaneChunkBytes, bytes - offset);
            CUDA_CHECK(cudaMemcpyAsync(lane.host[r] + offset, source + offset, n,
                                       cudaMemcpyDeviceToHost, send_[r]));
            CUDA_CHECK(cudaEventRecord(lane.chunk_done[r][c], send_[r]));
        }
    }
    // Leg 2, both ranks: the peer's host buffer -> own staging on the receive stream, chunk by
    // chunk behind the peer's events; the lane's completion is recorded right after its last
    // copy, so a later lane's copies on the same stream do not delay this lane's finish.
    for (int rank = 0; rank < 2; ++rank) {
        const auto r    = static_cast<std::size_t>(rank);
        const auto peer = static_cast<std::size_t>(1 - rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        auto* destination = static_cast<std::byte*>(staging[r].data);
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t offset = c * kLaneChunkBytes;
            const std::size_t n      = std::min(kLaneChunkBytes, bytes - offset);
            CUDA_CHECK(cudaStreamWaitEvent(receive_[r], lane.chunk_done[peer][c], 0));
            CUDA_CHECK(cudaMemcpyAsync(destination + offset, lane.host[peer] + offset, n,
                                       cudaMemcpyHostToDevice, receive_[r]));
        }
        CUDA_CHECK(cudaEventRecord(lane.received[r], receive_[r]));
    }
    lane.buffer  = buffer;
    lane.staging = staging;
    lane.active  = true;
}

void SplitLanes::sum_finish(int lane_index, const ExecutionContext& ec) {
    require(serves(ec), "SplitLanes: serves another device pair");
    require(lane_index >= 0 && lane_index < kCount, "SplitLanes: sum lane out of range");
    Lane& lane = lanes_[static_cast<std::size_t>(lane_index)];
    require(lane.active, "SplitLanes: sum_finish without sum_begin");
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        const cudaStream_t stream = ec.dev[rank]->stream;
        // Inbound data complete, and the peer has read rank r's host buffer (its receive copies
        // follow rank r's device-to-host reads of buffer[r]): buffer[r] may be overwritten and
        // every buffer of the lane reused.
        CUDA_CHECK(cudaStreamWaitEvent(stream, lane.received[r], 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream, lane.received[1 - r], 0));
        Tensor accumulator = lane.buffer[r];
        detail::residual_add_launch(lane.staging[r], accumulator, stream);
    }
    lane.buffer  = {};
    lane.staging = {};
    lane.active  = false;
}

bool allreduce_sum_split_available(const ExecutionContext& ec, const PeerEvents& events) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value() || !events.live()) {
        return false;
    }
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ec.dev[rank]->stream, &status));
        if (status != cudaStreamCaptureStatusActive) { continue; }
        return false;
    }
    return true;
}

void allreduce_sum_begin(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                         const ExecutionContext& ec, const PeerEvents& events, int lane) {
    detail::require_allreduce_sum_arguments(buffer, staging, ec, events);
    if (lane < 0 || lane >= SplitLanes::kCount) {
        throw std::invalid_argument("allreduce_sum_begin: sum lane out of range");
    }
    if (!allreduce_sum_split_available(ec, events)) {
        throw std::invalid_argument(
            "allreduce_sum_begin: needs an eager pair (both ranks' streams not capturing)");
    }
    events.ensure_split_lanes(ec).sum_begin(lane, buffer, staging, ec);
}

void allreduce_sum_finish(const ExecutionContext& ec, const PeerEvents& events, int lane) {
    SplitLanes* lanes = events.split_lanes();
    if (lanes == nullptr || !lanes->sum_open(lane)) {
        throw std::invalid_argument("allreduce_sum_finish: the lane has no open call");
    }
    lanes->sum_finish(lane, ec);
}

void detail::require_allreduce_sum_arguments(const std::array<Tensor, 2>& buffer,
                                             const std::array<Tensor, 2>& staging,
                                             const ExecutionContext& ec,
                                             const PeerEvents& events) {
    require_two_devices(ec,
                        "allreduce_sum: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < 2; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on both devices");
        }
    }
    require(events.live(), "allreduce_sum: events must be live");
#ifndef NDEBUG
    const std::size_t bytes = buffer[0].bytes();
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif
}

void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerEvents& events) {
    detail::require_allreduce_sum_arguments(buffer, staging, ec, events);
    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }

    const CurrentDeviceGuard guard;

    // MAILBOX TRANSPORT (see captured_mailbox()): one exchange kernel per device. Both ranks
    // publish their partial into their own pinned host slot, release an epoch flag, wait for the
    // peer's, and combine locally with the same arithmetic as residual_add_launch below.
    if (PeerMailbox* mailbox = captured_mailbox(buffer, bytes, ec, events)) {
        const int slot = mailbox->take_capture_slot();
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            mailbox->enqueue_exchange_sum(rank, slot, buffer[rank].data, bytes, local.stream);
        }
        return;
    }

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: each rank pulls the peer's operand into storage only it owns. Both inbound copies
    // are issued before either rank waits again, so the two directions overlap.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        Tensor accumulator = buffer[rank];
        detail::residual_add_launch(staging[rank], accumulator, local.stream);
    }
}

void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allgather_rows: requires an ExecutionContext with two distinct devices");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < 2; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    require(part[0].ne[1] + part[1].ne[1] == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live(), "allgather_rows: events must be live");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    const std::size_t block[2]  = {row_bytes * static_cast<std::size_t>(part[0].ne[1]),
                                   row_bytes * static_cast<std::size_t>(part[1].ne[1])};
    const std::size_t offset[2] = {0, block[0]};

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: rank r writes its own block locally and pulls the peer's block, both on its own
    // stream, so destination[r] has exactly one writer.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                   part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                   local.stream));
        CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[1 - rank]),
                             part[1 - rank].data, block[1 - rank], local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
    }
}

} // namespace ninfer::ops
