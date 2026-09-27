#pragma once

// ninfer::ops::detail - the kernels behind the pinned-host mailbox transport for TP2 collectives.
//
// TRANSPORT CONTRACT (why this exists). Without peer access a cross-device cudaMemcpyAsync is
// staged by the driver through host memory, and the event chain that orders the staged copy costs
// far more than the payload. An exchange kernel pair replaces the whole choreography: both ranks
// run one kernel CONCURRENTLY, publish their operand into their own pinned host slot, release a
// flag, spin on the peer's flag, then sum locally -- no events, no copy engine, no driver round
// trip inside the exchange. Measured on 2x RTX 5060 Ti under Windows 11 WDDM: ~41 us per 10 KiB
// reduction (graph-replayed) against ~277 us for the staged path.
//
// TWO KERNELS, ONE TRANSPORT. peer_exchange_pipelined_kernel (the default) and the original
// peer_exchange_sum_kernel (kept verbatim, selected by NINFER_TP_MAILBOX_LEGACY=1 for A/B runs on
// one binary) move the same bytes and combine them with the same arithmetic; they differ only in
// how the protocol is split across threads, so a mailbox runs exactly one of them for its whole
// life (PeerMailbox::kernel()) and the two never share a slab.
//
// The original kernel publishes the WHOLE payload behind one flag per slot: every block fences,
// meets the others on a device-scope arrival counter, and the last one releases; each thread then
// moves its four 16-byte vectors one at a time (the in-place stores keep the compiler from
// overlapping the next load with them), so a publish costs four dependent VRAM loads and a read
// four dependent PCIe round trips. The pipelined kernel makes each WARP an independent mailbox
// lane with its own flag and epoch, loads all its vectors before storing any, and overlaps its
// first probe of the peer's flag with its own publish; see the second EPOCH PROTOCOL below.
//
// EPOCH PROTOCOL, ORIGINAL KERNEL (per slot and rank; see PeerMailbox for how slots are assigned).
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

// Probes of __nanosleep(100) polling before the poller gives up and reports. One probe costs about
// 0.8 us (4M probes measured 3.25 s on two RTX 5070 Ti), so the limit is ~0.8 s: long enough for
// any legitimate skew between the two ranks' schedules, short enough to surface a missing peer
// before a 2 s display watchdog would reset the device (test_allreduce checks the bound).
inline constexpr std::uint32_t kPeerSpinLimit = 1000000u;

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

// ---- pipelined kernel (the default) ------------------------------------------------------------
//
// EPOCH PROTOCOL, PIPELINED KERNEL (per slot, per warp lane and rank). Warp w of the grid owns the
// vectors w*32*G + j*32 + lane, j in [0, G) -- 512*G contiguous bytes, every warp access 512
// contiguous bytes -- plus one release flag of its own in the slot (mine_flags[w *
// kPeerFlagStride], a 64-byte host line each) and one epoch of its own (epochs[w], this device's
// memory):
//
//   publish: lane 0 reads target = epochs[w] + 1; every lane loads its G partial vectors (all in
//            flight), then stores them to the rank's host slot; __syncwarp() orders the warp's
//            stores before lane 0's release, and lane 0 stores mine_flags[w] = target with release
//            semantics at system scope (one fence.acq_rel.sys, cumulative over the warp barrier).
//   consume: lane 0 polls peer_flags[w] with ACQUIRE system loads (ld.acquire.sys: no fence
//            instruction on this path, unlike the original kernel's relaxed polls plus fence)
//            until it reaches target, with __nanosleep between probes, and stores epochs[w] =
//            target; __syncwarp() hands the acquire to the warp, and every lane reads its G peer
//            vectors (ld.global.cv, all in flight) and combines them with the registers it
//            published from.
//
// Lockstep holds per (slot, warp) exactly as per slot above: both ranks run the same exchanges
// with the same geometry, so a warp lane that an exchange of a slot leaves unused is skipped by
// both ranks and its epoch and flags stay equal on both. SLOT REUSE holds as well: a rank starts
// exchange k + 2 only after its exchange k + 1 observed at least one of the peer's k + 1 flags,
// which the peer releases from its k + 1 kernel, launched after its whole exchange k kernel (every
// warp's read) finished. No warp waits on another warp or block of its own rank, so unlike the
// original kernel the pipelined one needs no co-residency of its blocks on either device, and the
// peer's reads of one warp's chunk start while later chunks are still being published. The HANG
// GUARD is the original one, per warp: a warp that gives up skips its own combine.
//
// ARITHMETIC. Identical to the original kernel: for every element float(mine) + float(peer) in
// FP32 and one __floats2bfloat162_rn, `mine` being exactly the value this rank published. The
// bits of the result do not depend on which kernel moved the operands.

// Words between two warps' release flags: a 64-byte host line each.
inline constexpr int kPeerFlagStride = 16;

// Sleep between two polls of the peer's flag; kPeerSpinLimit is calibrated with it.
inline constexpr std::uint32_t kPeerPollSleepNs = 100u;

// The production geometry: 16-byte vectors per lane and threads per block (see
// tools/tp2/mailbox_probe.cu --sweep for the measurements behind them).
inline constexpr int kPeerPipelinedVecsPerLane = 2;
inline constexpr int kPeerPipelinedThreads     = 256;

// Launch knobs of one pipelined exchange. Production passes the defaults; the standalone probe
// varies them and, with kTimed, collects every warp's phase timestamps.
struct PeerPipelinedTuning {
    std::uint32_t spin_limit   = kPeerSpinLimit;
    std::uint32_t sleep_ns     = kPeerPollSleepNs;
    unsigned long long* stamps = nullptr; // kTimed only: kPeerStampCount words per warp
};

// kTimed stamps, per warp (lane 0): globaltimer at entry, clock64 at entry, after the publish
// stores issued (the partial loads returned), after the release store issued (its system fence
// completed), when the peer's flag was seen (acquired), at the end (the peer loads returned and
// the combine stored), globaltimer at the end, and the number of polls that missed.
inline constexpr int kPeerStampCount = 8;

// Poll flavours of the pipelined kernel. Production polls with acquire loads after its release;
// the probe also measures an early relaxed first probe (issued before the publish, so its PCIe
// round trip overlaps it) followed by relaxed polls and one acquire fence, the original kernel's
// consume path.
inline constexpr int kPeerPollAcquire    = 0;
inline constexpr int kPeerPollEarlyFence = 1;

__device__ __forceinline__ unsigned long long peer_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One rank's half of the pipelined two-rank allreduce exchange; arguments as for
// peer_exchange_sum_kernel, except that `mine_flags`/`peer_flags` are the slot's first warp flag
// (warp w's at w * kPeerFlagStride) and `epochs` the slot's first warp epoch (warp w's at w).
// Launch geometry: peer_pipelined_blocks(vecs, G, threads) blocks of `threads` (a multiple of 32,
// at most 256), identical on both ranks for one slot execution.
template <int G, int kPoll = kPeerPollAcquire, bool kTimed = false>
__global__ __launch_bounds__(256) void peer_exchange_pipelined_kernel(
    PeerVecBf16* partial, PeerVecBf16* mine_payload, std::uint32_t* mine_flags,
    const PeerVecBf16* peer_payload, std::uint32_t* peer_flags, std::uint32_t* epochs,
    std::uint32_t* hang, int vecs, PeerPipelinedTuning tuning) {
    using SystemWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;

    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int base = warp * (32 * G) + lane;
    if (warp * (32 * G) >= vecs) { return; } // warp-uniform: the grid's last warps may be idle

    unsigned long long stamp[5] = {};
    const bool timed = kTimed && lane == 0 && tuning.stamps != nullptr;
    if (timed) {
        stamp[0] = peer_globaltimer();
        stamp[1] = clock64();
    }

    std::uint32_t target = 0;
    std::uint32_t seen   = 0;
    if (lane == 0) {
        target = epochs[warp] + 1u;
        if constexpr (kPoll == kPeerPollEarlyFence) {
            seen = SystemWord(peer_flags[warp * kPeerFlagStride]).load(cuda::memory_order_relaxed);
        }
    }

    // Publish: every load in flight before the first store. Copies go through the trivial `raw`
    // member (see peer_exchange_sum_kernel).
    PeerVec mine[G] = {};
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { mine[j] = partial[v].raw; }
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { mine_payload[v].raw = mine[j]; }
    }
    __syncwarp();
    if (timed) { stamp[2] = clock64(); }

    int published       = 1;
    std::uint32_t spins = 0;
    if (lane == 0) {
        SystemWord(mine_flags[warp * kPeerFlagStride]).store(target, cuda::memory_order_release);
        if (timed) { stamp[3] = clock64(); }
        const SystemWord peer(peer_flags[warp * kPeerFlagStride]);
        const SystemWord fault(*hang);
        if constexpr (kPoll == kPeerPollAcquire) {
            seen = peer.load(cuda::memory_order_acquire);
        }
        while (static_cast<std::int32_t>(seen - target) < 0) {
            __nanosleep(tuning.sleep_ns);
            ++spins;
            if (spins % kPeerHangProbe == 0u &&
                (spins > tuning.spin_limit || fault.load(cuda::memory_order_relaxed) != 0u)) {
                fault.store(1u, cuda::memory_order_relaxed);
                published = 0;
                break;
            }
            seen = peer.load(kPoll == kPeerPollAcquire ? cuda::memory_order_acquire
                                                       : cuda::memory_order_relaxed);
        }
        if constexpr (kPoll == kPeerPollEarlyFence) {
            // Acquire: the peer payload reads below are ordered after the observed release.
            cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
        }
        epochs[warp] = target;
        if (timed) { stamp[4] = clock64(); }
    }
    __syncwarp();
    if (__shfl_sync(0xffffffffu, published, 0) == 0) { return; }

    // Consume: every peer load in flight, then combine in place with the published registers.
    PeerVec peer[G] = {};
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { peer[j] = __ldcv(&peer_payload[v].raw); }
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v >= vecs) { continue; }
        PeerVecBf16 sum;
        PeerVecBf16 other;
        sum.raw   = mine[j];
        other.raw = peer[j];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float a0 = __low2float(sum.pair[pair]);
            const float b0 = __high2float(sum.pair[pair]);
            const float a1 = __low2float(other.pair[pair]);
            const float b1 = __high2float(other.pair[pair]);
            sum.pair[pair] = __floats2bfloat162_rn(a0 + a1, b0 + b1);
        }
        partial[v].raw = sum.raw;
    }
    if (timed) {
        const unsigned long long end = clock64();
        unsigned long long* out = tuning.stamps + static_cast<std::size_t>(warp) * kPeerStampCount;
        out[0]                  = stamp[0];
        out[1]                  = stamp[1];
        out[2]                  = stamp[2];
        out[3]                  = stamp[3];
        out[4]                  = stamp[4];
        out[5]                  = end;
        out[6]                  = peer_globaltimer();
        out[7]                  = spins;
    }
}

// Warp lanes one pipelined exchange of `vecs` vectors uses, at `vecs_per_lane` vectors per lane.
inline int peer_pipelined_warps(std::size_t vecs, int vecs_per_lane) {
    const std::size_t per_warp = 32u * static_cast<std::size_t>(vecs_per_lane);
    const std::size_t warps    = (vecs + per_warp - 1) / per_warp;
    return warps < 1 ? 1 : static_cast<int>(warps);
}

// Blocks of `threads` for one pipelined exchange of `vecs` vectors.
inline int peer_pipelined_blocks(std::size_t vecs, int vecs_per_lane, int threads) {
    const int warps_per_block = threads / 32;
    return (peer_pipelined_warps(vecs, vecs_per_lane) + warps_per_block - 1) / warps_per_block;
}

} // namespace ninfer::ops::detail
