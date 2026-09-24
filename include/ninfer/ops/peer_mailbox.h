#pragma once

// ninfer::ops - pinned-host mailbox transport for the two-device TP2 collectives.
//
// A PeerMailbox owns the pinned host memory and the per-device words the mailbox exchange (see
// src/ops/kernel/peer_exchange.cuh) runs on. Attached to a PeerEvents instance
// (PeerEvents::attach_mailbox), it becomes the transport of that stream pair's CAPTURED
// allreduce_sum calls whose payload fits a slot: one exchange kernel per device instead of the
// staged path's two event-ordered cross-device copies and two combines. Eager calls, oversized
// payloads and allgather_rows stay on the staged path, whose results the mailbox reproduces bit
// for bit.
//
// MEASURED ON A TRANSPORT-DEGRADED PAIR (2x RTX 5060 Ti, Windows 11 WDDM, no P2P):
//   staged event path (graph replay):   ~277 us per 10 KiB reduction
//   mailbox exchange (graph replay):     ~41 us per 10 KiB reduction
//
// LIFECYCLE. The owner (a Program) creates the mailbox once, before capturing any graph that
// uses it (cudaHostAlloc and cudaMalloc are neither stream-ordered nor capturable), attaches it
// to its PeerEvents, and destroys it only after every graph executable captured with it: the
// captured kernels bake in the slab and word addresses.
//
// SLOTS. Captured exchanges take slots round robin (take_capture_slot()). The epoch protocol
// makes two slots sufficient for any sequence of captured exchanges and graph launches (see
// peer_exchange.cuh, SLOT REUSE), so the slab never runs out and repeated or abandoned captures
// leak nothing. The only preconditions are those of every captured decode graph: its launches
// are issued on one stream, and every launch executes each captured exchange on both devices.
//
// FAULTS. A poller that waits too long for its peer (kPeerSpinLimit, qualified below the 2 s display
// watchdog by test_allreduce) gives up, skips its combine and sets a sticky hang word;
// hang_reported() exposes it to the owner, which fails that round and every later one.

#include "core/device.h" // ExecutionContext

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

class PeerMailbox {
public:
    // Allocates the pinned host slab (`slots` payload slots of `slot_bytes`, rounded up to 256
    // bytes, and one release word per slot, per rank, plus the hang word) and each device's
    // arrival and epoch words, and loads the exchange kernel on both devices so that no capture
    // has to. Throws std::invalid_argument for a context without two distinct devices, fewer than
    // two slots or an empty slot, and std::runtime_error on allocation failure.
    PeerMailbox(const ExecutionContext& ec, std::size_t slot_bytes, int slots = 2);
    ~PeerMailbox();

    PeerMailbox(const PeerMailbox&)            = delete;
    PeerMailbox& operator=(const PeerMailbox&) = delete;
    PeerMailbox(PeerMailbox&&)                 = delete;
    PeerMailbox& operator=(PeerMailbox&&)      = delete;

    // True when this mailbox was created for `ec`'s device pair, in rank order.
    [[nodiscard]] bool serves(const ExecutionContext& ec) const noexcept;

    // The largest payload one exchange carries.
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }

    // The slot of the next captured exchange, round robin. Called once per captured call site,
    // for both ranks together.
    [[nodiscard]] int take_capture_slot() noexcept;

    // Enqueues rank `rank`'s half of the summing exchange of `bytes` (a multiple of 16, at most
    // slot_bytes()) at the 16-byte aligned `data`, resident on that rank's device, onto `stream`.
    // The caller issues both ranks' halves for one slot, with the same `bytes`, inside one
    // capture, and makes `rank`'s device current.
    void enqueue_exchange_sum(int rank, int slot, void* data, std::size_t bytes,
                              cudaStream_t stream) const;

    // True once an exchange's poller gave up waiting for its peer. Sticky: the two ranks' results
    // diverged, so the owner must not trust any later round.
    [[nodiscard]] bool hang_reported() const noexcept;

private:
    void* slab_                     = nullptr; // pinned host allocation, UVA-mapped
    std::uint8_t* payload_[2]       = {nullptr, nullptr};
    std::uint32_t* flags_[2]        = {nullptr, nullptr};
    std::uint32_t* hang_            = nullptr;
    std::uint32_t* device_words_[2] = {nullptr, nullptr}; // [arrival x slots][epoch x slots]
    std::size_t slot_bytes_         = 0;
    int slots_                      = 0;
    int next_slot_                  = 0;
    int devices_[2]                 = {0, 0};
};

} // namespace ninfer::ops
