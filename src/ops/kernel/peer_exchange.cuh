#pragma once

// ninfer::ops::detail - the kernel behind the pinned-host mailbox transport for TP2 collectives.
//
// TRANSPORT CONTRACT (why this exists). Without peer access a cross-device cudaMemcpyAsync is
// staged by the driver through host memory, and the event chain that orders the staged copy costs
// far more than the payload. This kernel pair replaces the whole choreography: both ranks run one
// kernel CONCURRENTLY, publish their operand into their own pinned host slot, release a flag,
// spin on the peer's flag, then sum locally -- no events, no copy engine, no driver round trip
// inside the exchange. Measured on 2x RTX 5060 Ti under Windows 11 WDDM: ~41 us per 10 KiB
// reduction (graph-replayed) against ~277 us for the staged path.
//
// EPOCH PROTOCOL (per slot and rank; see PeerMailbox for how slots are assigned).
//
//   every block, before it arrives:   target = epoch + 1     (epoch: this rank's count of
//                                                              completed publishes of this slot,
//                                                              in this device's memory)
//   publish:  every thread stores its 16-byte payload chunks to the rank's host slot
//             __threadfence_system()          -- payload stores leave this GPU for system memory
//             __syncthreads()                 -- the block meets
//             thread 0: arrival.fetch_add(1)  -- acq_rel, device scope
//             the block that completes the arrival count (every block fenced and arrived):
//                 arrival = 0, epoch = target
//                 mine_flag = target          -- release, system scope
//   consume:  thread 0 spins until peer_flag >= target (relaxed system loads), then an acquire
//             fence; __syncthreads() hands the acquire to the whole block, and every thread reads
//             its peer payload chunks (ld.global.cv) and combines.
//
// Both ranks execute the same schedule, so each slot's epochs advance in lockstep: execution n of
// a slot publishes n on both ranks and waits for n from the peer. A flag left at n - 1 by the
// previous execution never satisfies the wait, so the protocol needs no host reset between graph
// replays and no host synchronization between launches. (A 0/1 flag would: without a reset the
// second replay would see the first replay's 1 and read a payload the peer has not republished.)
// Comparisons are wrap-safe.
//
// SLOT REUSE. A rank reaches exchange k + 2 only after it observed the peer's publish of k + 1,
// which the peer issues after its exchange k finished reading. Consecutive exchanges on two
// alternating slots therefore never overwrite a payload or flag the peer still reads, and
// exchanges of successive graph launches are separated by the launches themselves (a graph
// launch completes on both devices before the next launch on the same stream starts).
//
// HANG GUARD. A poller gives up after kPeerSpinLimit probes, or earlier once another exchange
// already reported a hang, sets the pinned hang word and skips its combine. The Program reads the
// word after every round's device synchronization and fails the round (PeerMailbox::
// hang_reported()); the word is sticky, because the ranks' results have diverged.
//
// ARITHMETIC. The combine is the qualified residual_add body: FP32 accumulation of the two
// represented BF16 operands, one round-to-nearest-even on store. This matches the staged path's
// local combine bit for bit (the same two partials, the same order, the same rounding), so the
// mailbox transport cannot change a single output value.

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// ~0.4 s of __nanosleep(100) polling before the poller gives up and reports: long enough for any
// legitimate skew between the two ranks' schedules, short enough to surface a missing peer well
// before a display watchdog would reset the device.
inline constexpr std::uint32_t kPeerSpinLimit = 4000000u;

// A poller re-reads the pinned hang word once every this many probes, so a round that already
// reported a hang does not spin the full limit again at every later exchange.
inline constexpr std::uint32_t kPeerHangProbe = 1024u;

using PeerVec = uint4; // 16 bytes = 8 BF16 elements, one PCIe transaction per access

// 16-byte chunks per thread per pass. A warp then touches 4 consecutive 512 B lines of the
// payload: 64 B per thread across a 2 KB span, so both the host write burst and the read phase
// coalesce into 128 B PCIe transactions instead of per-16 B ones.
inline constexpr int kPeerGroup = 4;

union PeerVecBf16 {
    PeerVec raw;
    __nv_bfloat162 pair[4];
};

// One rank's half of the two-rank allreduce exchange.
//
//   partial         this rank's operand, replaced in place by the sum (this device's VRAM)
//   mine_payload    this rank's pinned host publish slot; only this kernel writes it
//   mine_flag       this rank's pinned release word for the slot
//   peer_payload    the peer's publish slot, written by the OTHER GPU while this kernel runs,
//                   hence neither const-restrict nor read through the non-coherent path
//   peer_flag       the peer's release word
//   epoch           this slot's publish count for this rank (this device's VRAM)
//   arrival         this slot's block-arrival counter for this rank (this device's VRAM)
//   hang            the pinned aggregate hang word (PeerMailbox::hang_reported())
//   vecs            payload length in 16-byte units
//
// Launch geometry: any grid, 256 threads, identical on both ranks for one slot execution. For
// the 10 KiB decode activation (640 vectors) the launcher picks 1 block; wider payloads scale to
// more blocks so the read phase can overlap PCIe latency across SMs.
__global__ __launch_bounds__(256) void peer_exchange_sum_kernel(
    PeerVecBf16* partial, PeerVecBf16* mine_payload, std::uint32_t* mine_flag,
    const PeerVecBf16* peer_payload, std::uint32_t* peer_flag, std::uint32_t* epoch,
    std::uint32_t* arrival, std::uint32_t* hang, int vecs) {
    using DeviceWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>;
    using SystemWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;

    __shared__ int combine;

    const int tid        = blockIdx.x * blockDim.x + threadIdx.x;
    const int group_span = gridDim.x * blockDim.x * kPeerGroup;

    // Read before this block arrives: the last block to arrive advances `epoch`, and the arrival
    // counter's acq_rel ordering keeps that update invisible to this read.
    std::uint32_t target = 0;
    if (threadIdx.x == 0) { target = DeviceWord(*epoch).load(cuda::memory_order_relaxed) + 1u; }

    // Publish this rank's operand into its pinned host slot. Copies go through the trivial
    // `raw` member: the union's BF16 members have non-trivial special members, which would make
    // the whole union non-copyable in device code.
    for (int g = tid * kPeerGroup; g < vecs; g += group_span) {
#pragma unroll
        for (int j = 0; j < kPeerGroup; ++j) {
            if (g + j < vecs) { mine_payload[g + j].raw = partial[g + j].raw; }
        }
    }
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        // Every block fences before it arrives, so the last arrival implies every payload store
        // from every block is already system-visible.
        const std::uint32_t arrived =
            DeviceWord(*arrival).fetch_add(1u, cuda::memory_order_acq_rel);
        if (arrived == gridDim.x - 1u) {
            DeviceWord(*arrival).store(0u, cuda::memory_order_relaxed);
            DeviceWord(*epoch).store(target, cuda::memory_order_relaxed);
            SystemWord(*mine_flag).store(target, cuda::memory_order_release);
        }

        const SystemWord peer(*peer_flag);
        const SystemWord fault(*hang);
        bool published      = true;
        std::uint32_t spins = 0;
        while (static_cast<std::int32_t>(peer.load(cuda::memory_order_relaxed) - target) < 0) {
            __nanosleep(100);
            ++spins;
            if (spins % kPeerHangProbe == 0u &&
                (spins > kPeerSpinLimit || fault.load(cuda::memory_order_relaxed) != 0u)) {
                fault.store(1u, cuda::memory_order_relaxed);
                published = false;
                break;
            }
        }
        // Acquire: the peer payload reads below are ordered after the observed release.
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
        combine = published ? 1 : 0;
    }
    __syncthreads();
    if (combine == 0) { return; }

    // Combine in place: FP32 accumulate of the two represented BF16 operands, single
    // round-to-nearest-even on store. Each thread reads exactly the chunks it published, so the
    // in-place update never races another thread's read.
    for (int g = tid * kPeerGroup; g < vecs; g += group_span) {
#pragma unroll
        for (int j = 0; j < kPeerGroup; ++j) {
            if (g + j >= vecs) { break; }
            PeerVecBf16 mine;
            PeerVecBf16 peer;
            mine.raw = partial[g + j].raw;
            peer.raw = __ldcv(&peer_payload[g + j].raw);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float a0  = __low2float(mine.pair[pair]);
                const float b0  = __high2float(mine.pair[pair]);
                const float a1  = __low2float(peer.pair[pair]);
                const float b1  = __high2float(peer.pair[pair]);
                mine.pair[pair] = __floats2bfloat162_rn(a0 + a1, b0 + b1);
            }
            partial[g + j].raw = mine.raw;
        }
    }
}

// Launch geometry for a payload of `bytes`: one block per 1024 vectors (16 KiB), at most 16, so the
// 10 KiB single-token decode activation runs as one block and the 40 KiB MTP-3 verify as three.
inline int peer_exchange_blocks(std::size_t bytes) {
    const std::size_t vecs = bytes / sizeof(PeerVec);
    const std::size_t want = (vecs + kPeerGroup * 256 - 1) / (kPeerGroup * 256);
    return want < 1 ? 1 : (want > 16 ? 16 : static_cast<int>(want));
}

} // namespace ninfer::ops::detail
