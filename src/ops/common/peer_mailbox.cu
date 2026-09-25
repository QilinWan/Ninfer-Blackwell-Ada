// Implements: include/ninfer/ops/peer_mailbox.h
//
// The pinned host slab layout, in one allocation:
//
//   [ payload rank0 slot0 .. slotN-1 ][ payload rank1 slot0 .. slotN-1 ]
//   [ flags rank0 slot0..N-1 ][ flags rank1 slot0..N-1 ][ hang word ]
//
// Payload slots are 256-byte aligned (the vectorized exchange reads and writes 16-byte units;
// the headroom also keeps a slot's stores on distinct cache lines). Flag words sit after the
// payload so a payload overflow from a mis-sized slot cannot reach them without being loudly
// out of contract, and the hang word is last.
//
// Each rank's arrival counters and epochs live in that rank's device memory: only that rank's
// kernels touch them, and the epoch is read at the start of every exchange, where a pinned host
// read would add a PCIe round trip.

#include "ninfer/ops/peer_mailbox.h"

#include "ops/kernel/peer_exchange.cuh" // detail::peer_exchange_sum_kernel

#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr std::size_t kSlotAlignment = 256;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kSlotAlignment - 1) & ~(kSlotAlignment - 1);
}

// Current-device save/restore for the phases that need a rank's device current.
class ScopedDevice {
public:
    ScopedDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void require_two_devices(const ExecutionContext& ec, const char* message) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value() ||
        ec.dev[0]->device == ec.dev[1]->device) {
        throw std::invalid_argument(message);
    }
}

void throw_on_error(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("PeerMailbox: ") + what + " failed: " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

PeerMailbox::PeerMailbox(const ExecutionContext& ec, std::size_t slot_bytes, int slots)
    : slot_bytes_(aligned(slot_bytes)), slots_(slots) {
    require_two_devices(ec, "PeerMailbox: requires an ExecutionContext with two distinct devices");
    if (slots < 2) { throw std::invalid_argument("PeerMailbox: requires at least two slots"); }
    if (slot_bytes == 0) { throw std::invalid_argument("PeerMailbox: slot bytes must be nonzero"); }

    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    devices_[0]       = pair[0];
    devices_[1]       = pair[1];

    const std::size_t payload_bytes = static_cast<std::size_t>(slots_) * slot_bytes_ * 2;
    const std::size_t words_bytes =
        (static_cast<std::size_t>(slots_) * 2 + 1) * sizeof(std::uint32_t);
    const std::size_t device_words_bytes =
        static_cast<std::size_t>(slots_) * 2 * sizeof(std::uint32_t);

    // Everything allocates into locals first and commits to members only when the whole set
    // succeeded: a throwing constructor does not run the destructor, so a half-built object
    // must leave nothing behind that needs it.
    void* slab                 = nullptr;
    std::uint32_t* words[2]    = {nullptr, nullptr};
    const ScopedDevice scope;
    try {
        throw_on_error(cudaHostAlloc(&slab, payload_bytes + words_bytes,
                                     cudaHostAllocMapped | cudaHostAllocPortable),
                       "cudaHostAlloc");
        std::memset(slab, 0, payload_bytes + words_bytes);
        for (int rank = 0; rank < 2; ++rank) {
            ScopedDevice::set(pair[rank]);
            throw_on_error(cudaMalloc(&words[rank], device_words_bytes), "cudaMalloc");
            // The legacy default stream does not order the Program's non-blocking streams, so the
            // zeroing is retired here rather than left for the first captured exchange to race.
            throw_on_error(cudaMemsetAsync(words[rank], 0, device_words_bytes, nullptr),
                           "cudaMemsetAsync");
            throw_on_error(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
            // Loads the exchange kernel on this device now: a module cannot be loaded inside the
            // capture that first launches it.
            cudaFuncAttributes attributes{};
            throw_on_error(cudaFuncGetAttributes(&attributes, detail::peer_exchange_sum_kernel),
                           "cudaFuncGetAttributes");
        }
    } catch (...) {
        if (slab != nullptr) { (void)cudaFreeHost(slab); }
        for (int rank = 0; rank < 2; ++rank) {
            if (words[rank] != nullptr) {
                (void)cudaSetDevice(pair[rank]);
                (void)cudaFree(words[rank]);
            }
        }
        throw;
    }

    slab_            = slab;
    auto* base       = static_cast<std::uint8_t*>(slab_);
    payload_[0]      = base;
    payload_[1]      = base + static_cast<std::size_t>(slots_) * slot_bytes_;
    auto* flags      = reinterpret_cast<std::uint32_t*>(base + payload_bytes);
    flags_[0]        = flags;
    flags_[1]        = flags + slots_;
    hang_            = flags + 2 * static_cast<std::size_t>(slots_);
    device_words_[0] = words[0];
    device_words_[1] = words[1];
}

PeerMailbox::~PeerMailbox() {
    const ScopedDevice scope;
    // The owner destroys every graph executable first, but a launch may still be in flight: the
    // pinned slab and the counters are released only once both devices retired it.
    for (int rank = 0; rank < 2; ++rank) {
        if (cudaSetDevice(devices_[rank]) != cudaSuccess) { continue; }
        const cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaDeviceSynchronize: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
        (void)cudaFree(device_words_[rank]);
    }
    (void)cudaFreeHost(slab_);
}

bool PeerMailbox::serves(const ExecutionContext& ec) const noexcept {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    return ec.dev[0]->device == devices_[0] && ec.dev[1]->device == devices_[1];
}

int PeerMailbox::take_capture_slot() noexcept {
    const int slot = next_slot_;
    next_slot_     = (next_slot_ + 1) % slots_;
    return slot;
}

void PeerMailbox::enqueue_exchange_sum(int rank, int slot, void* data, std::size_t bytes,
                                       cudaStream_t stream) const {
    if (rank < 0 || rank > 1 || slot < 0 || slot >= slots_) {
        throw std::invalid_argument("PeerMailbox: rank or slot out of range");
    }
    if (bytes == 0 || bytes > slot_bytes_ || (bytes % sizeof(detail::PeerVec)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(data) % sizeof(detail::PeerVec)) != 0) {
        throw std::invalid_argument(
            "PeerMailbox: an exchange carries whole, aligned 16-byte vectors within one slot");
    }
    const std::size_t offset = static_cast<std::size_t>(slot) * slot_bytes_;
    std::uint32_t* words     = device_words_[rank];
    detail::peer_exchange_sum_kernel<<<detail::peer_exchange_blocks(bytes), 256, 0, stream>>>(
        static_cast<detail::PeerVecBf16*>(data),
        reinterpret_cast<detail::PeerVecBf16*>(payload_[rank] + offset), flags_[rank] + slot,
        reinterpret_cast<const detail::PeerVecBf16*>(payload_[1 - rank] + offset),
        flags_[1 - rank] + slot, words + slots_ + slot, words + slot, hang_,
        static_cast<int>(bytes / sizeof(detail::PeerVec)));
    CUDA_CHECK(cudaGetLastError());
}

bool PeerMailbox::hang_reported() const noexcept {
    return *static_cast<volatile const std::uint32_t*>(hang_) != 0;
}

void PeerMailbox::report_hang() noexcept { *static_cast<volatile std::uint32_t*>(hang_) = 1u; }

} // namespace ninfer::ops
