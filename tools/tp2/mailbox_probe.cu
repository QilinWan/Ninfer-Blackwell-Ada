// Standalone TP2 mailbox probe: does the pinned-host mailbox exchange work between these two GPUs,
// and what does one exchange cost?
//
// WHAT IT ANSWERS. NInfer's --tp 2 all-reduces have two transports. The MAILBOX runs one kernel
// per GPU concurrently: each publishes its operand into its own pinned host slot, releases a flag,
// spins on the peer's flag, reads the peer's slot and sums locally (no events, no copy engine,
// no driver round trip). The COPIES path stages cross-device cudaMemcpyAsync copies through the
// driver between event hops. The mailbox is faster, but it needs the two GPUs to observe each
// other's system-scope stores promptly; under some GPU virtualization (WSL2, issue #1) the flag
// never arrives and the exchange times out. The engine probes this at startup (one 4 KiB exchange
// captured in a two-device graph; timed out or slower than 50 ms => copies); this tool runs the
// same check without the engine or a model, plus the timings behind the choice, so a machine can
// be diagnosed in seconds and the report pasted into an issue.
//
// The exchange kernels are the production ones: this file includes
// src/ops/kernel/peer_exchange.cuh (CUDA headers only), so build it from a checkout. The mailbox
// is allocated as PeerMailbox does (pinned, mapped, portable) and, like the engine, alternates two
// slots. The default kernel is the pipelined one; --legacy selects the original kernel (the
// engine's NINFER_TP_MAILBOX_LEGACY=1). No engine library: nvcc and the CUDA runtime only.
//
// Build (from the repository root):
//   Linux    nvcc -O2 -std=c++20 -arch=sm_120a -o build/mailbox_probe tools/tp2/mailbox_probe.cu
//   Windows  nvcc -O2 -std=c++20 -arch=sm_120a -o mailbox_probe.exe tools\tp2\mailbox_probe.cu
//   (-arch=native also works on CUDA 12.8+; the exchange needs no sm_120-specific feature.)
// Run:
//   mailbox_probe [dev_a dev_b] [--payload BYTES] [--chain N] [--spin-limit N] [--legacy]
//                 [--simulate-hang] [--sweep] [--timed] [--work US] [--replays N]
//     dev_a dev_b     CUDA device ids (default 0 1)
//     --payload       exchange payload in bytes, multiple of 16 (default 10240, the decode activation)
//     --chain         exchanges per chain (default 128, one decode step's worth)
//     --spin-limit    pipelined poller probes before an exchange reports a hang (default 1000000,
//                     ~0.8 s; the original kernel's limit is the compiled kPeerSpinLimit)
//     --legacy        the original exchange kernel instead of the pipelined one
//     --simulate-hang rank 1 skips the startup exchange, to exercise the hang report path
//     --sweep         maintainer: µs per exchange for every kernel variant (1/2/4/8 vectors per lane,
//                     64/128/256 threads, both poll flavours, sleeps) at 4, 10 and 40 KiB
//     --timed         maintainer: per-phase anatomy of one exchange (clock64 stamps), both kernels
//     --work US       maintainer: a US-microsecond spin kernel on both ranks before every exchange
//                     (a stand-in for the layer between two all-reduces); reported costs exclude it
//     --replays N     graph replays per measurement (default 20)
// Exit status: 0 mailbox usable · 2 mailbox unusable (timed out or too slow) while the copies path
// works · 1 CUDA error or wrong sums · 77 fewer than two devices.
#include "../../src/ops/kernel/peer_exchange.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CK(expr)                                                                                 \
    do {                                                                                         \
        const cudaError_t err_ = (expr);                                                         \
        if (err_ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "%s:%d: %s failed: %s: %s\n", __FILE__, __LINE__, #expr,        \
                         cudaGetErrorName(err_), cudaGetErrorString(err_));                      \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

namespace {

namespace pd = ninfer::ops::detail;
using pd::PeerVec;
using pd::PeerVecBf16;

// ---- the original kernel with phase stamps (maintainer anatomy only) ---------------------------
// peer_exchange_sum_kernel line for line, plus clock64 stamps by thread 0 of block 0: globaltimer
// and clock at entry, after the publish loop, after __threadfence_system, after the arrival (and
// release), when the peer's flag was seen, after the acquire fence, after the second barrier, at
// the end, globaltimer at the end; word kLegacyStamps counts the polls that missed. The stamps
// only read clocks; the protocol and the memory operations are the production ones.
constexpr int kLegacyStamps = 10;

__global__ __launch_bounds__(256) void legacy_timed_kernel(
    PeerVecBf16* partial, PeerVecBf16* mine_payload, std::uint32_t* mine_flag,
    const PeerVecBf16* peer_payload, std::uint32_t* peer_flag, std::uint32_t* epoch,
    std::uint32_t* arrival, std::uint32_t* hang, int vecs, unsigned long long* stamps) {
    using DeviceWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>;
    using SystemWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;
    __shared__ int combine;
    const bool timed = blockIdx.x == 0 && threadIdx.x == 0;
    unsigned long long st[kLegacyStamps] = {};
    if (timed) { st[0] = pd::peer_globaltimer(); st[1] = clock64(); }
    const int tid        = blockIdx.x * blockDim.x + threadIdx.x;
    const int group_span = gridDim.x * blockDim.x * pd::kPeerGroup;
    std::uint32_t target = 0;
    if (threadIdx.x == 0) { target = DeviceWord(*epoch).load(cuda::memory_order_relaxed) + 1u; }
    for (int g = tid * pd::kPeerGroup; g < vecs; g += group_span) {
#pragma unroll
        for (int j = 0; j < pd::kPeerGroup; ++j) {
            if (g + j < vecs) { mine_payload[g + j].raw = partial[g + j].raw; }
        }
    }
    if (timed) { st[2] = clock64(); }
    __threadfence_system();
    if (timed) { st[3] = clock64(); }
    __syncthreads();
    std::uint32_t spins = 0;
    if (threadIdx.x == 0) {
        const std::uint32_t arrived = DeviceWord(*arrival).fetch_add(1u, cuda::memory_order_acq_rel);
        if (arrived == gridDim.x - 1u) {
            DeviceWord(*arrival).store(0u, cuda::memory_order_relaxed);
            DeviceWord(*epoch).store(target, cuda::memory_order_relaxed);
            SystemWord(*mine_flag).store(target, cuda::memory_order_release);
        }
        if (timed) { st[4] = clock64(); }
        const SystemWord peer(*peer_flag);
        const SystemWord fault(*hang);
        bool published = true;
        while (static_cast<std::int32_t>(peer.load(cuda::memory_order_relaxed) - target) < 0) {
            __nanosleep(100);
            ++spins;
            if (spins % pd::kPeerHangProbe == 0u &&
                (spins > pd::kPeerSpinLimit || fault.load(cuda::memory_order_relaxed) != 0u)) {
                fault.store(1u, cuda::memory_order_relaxed);
                published = false;
                break;
            }
        }
        if (timed) { st[5] = clock64(); }
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
        if (timed) { st[6] = clock64(); }
        combine = published ? 1 : 0;
    }
    __syncthreads();
    if (timed) { st[7] = clock64(); }
    if (combine == 0) { return; }
    for (int g = tid * pd::kPeerGroup; g < vecs; g += group_span) {
#pragma unroll
        for (int j = 0; j < pd::kPeerGroup; ++j) {
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
    if (timed) {
        st[8] = clock64();
        for (int i = 0; i < 9; ++i) { stamps[i] = st[i]; }
        stamps[9]             = pd::peer_globaltimer();
        stamps[kLegacyStamps] = spins;
    }
}

// A spin of `ns` nanoseconds on one thread: the layer between two all-reduces, as far as the
// exchange can tell.
__global__ void spin_kernel(unsigned long long ns) {
    if (threadIdx.x != 0) { return; }
    const unsigned long long start = pd::peer_globaltimer();
    while (pd::peer_globaltimer() - start < ns) {}
}

// The copies path's local combine: partial += staged (same arithmetic as the exchange).
__global__ void add_bf16_kernel(PeerVecBf16* partial, const PeerVecBf16* staged, int vecs) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= vecs) { return; }
    PeerVecBf16 mine;
    PeerVecBf16 peer;
    mine.raw = partial[g].raw;
    peer.raw = staged[g].raw;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        mine.pair[pair] = __floats2bfloat162_rn(__low2float(mine.pair[pair]) + __low2float(peer.pair[pair]),
                                                __high2float(mine.pair[pair]) + __high2float(peer.pair[pair]));
    }
    partial[g].raw = mine.raw;
}

// ---- kernel variants ---------------------------------------------------------------------------
enum class Kind { Legacy, LegacyTimed, Pipelined };

struct Variant {
    Kind kind            = Kind::Pipelined;
    int vecs_per_lane    = pd::kPeerPipelinedVecsPerLane;
    int threads          = pd::kPeerPipelinedThreads;
    int poll             = pd::kPeerPollAcquire;
    std::uint32_t sleep  = pd::kPeerPollSleepNs;
    bool timed           = false;

    std::string name() const {
        if (kind != Kind::Pipelined) { return kind == Kind::Legacy ? "legacy" : "legacy-timed"; }
        char buffer[96];
        std::snprintf(buffer, sizeof buffer, "pipelined g%d t%d %s sleep%u%s", vecs_per_lane, threads,
                      poll == pd::kPeerPollAcquire ? "acquire" : "early+fence", sleep,
                      timed ? " timed" : "");
        return buffer;
    }
    int blocks(std::size_t vecs) const {
        if (kind != Kind::Pipelined) { return pd::peer_exchange_blocks(vecs * sizeof(PeerVec)); }
        return pd::peer_pipelined_blocks(vecs, vecs_per_lane, threads);
    }
};

template <int G, int P, bool T>
void launch_pipelined(const Variant& v, int blocks, cudaStream_t stream, PeerVecBf16* partial,
                      PeerVecBf16* mine, std::uint32_t* mine_flags, const PeerVecBf16* peer,
                      std::uint32_t* peer_flags, std::uint32_t* epochs, std::uint32_t* hang, int vecs,
                      pd::PeerPipelinedTuning tuning) {
    pd::peer_exchange_pipelined_kernel<G, P, T><<<blocks, v.threads, 0, stream>>>(
        partial, mine, mine_flags, peer, peer_flags, epochs, hang, vecs, tuning);
}

template <int G>
void launch_pipelined_g(const Variant& v, int blocks, cudaStream_t s, PeerVecBf16* a, PeerVecBf16* b,
                        std::uint32_t* c, const PeerVecBf16* d, std::uint32_t* e, std::uint32_t* f,
                        std::uint32_t* h, int n, pd::PeerPipelinedTuning t) {
    if (v.poll == pd::kPeerPollAcquire) {
        if (v.timed) { launch_pipelined<G, pd::kPeerPollAcquire, true>(v, blocks, s, a, b, c, d, e, f, h, n, t); }
        else { launch_pipelined<G, pd::kPeerPollAcquire, false>(v, blocks, s, a, b, c, d, e, f, h, n, t); }
    } else {
        if (v.timed) { launch_pipelined<G, pd::kPeerPollEarlyFence, true>(v, blocks, s, a, b, c, d, e, f, h, n, t); }
        else { launch_pipelined<G, pd::kPeerPollEarlyFence, false>(v, blocks, s, a, b, c, d, e, f, h, n, t); }
    }
}

// ---- host side ---------------------------------------------------------------------------------
double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct Options {
    int device[2]            = {0, 1};
    std::size_t payload      = 10240;
    int chain                = 128;
    std::uint32_t spin_limit = pd::kPeerSpinLimit;
    bool legacy              = false;
    bool simulate_hang       = false;
    bool sweep               = false;
    bool timed               = false;
    double work_us           = 0.0;
    int replays              = 20;
};

Options parse(int argc, char** argv) {
    Options o;
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value            = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", arg.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (arg == "--payload") {
            o.payload = std::strtoull(value(), nullptr, 10);
        } else if (arg == "--chain") {
            o.chain = std::atoi(value());
        } else if (arg == "--spin-limit") {
            o.spin_limit = static_cast<std::uint32_t>(std::strtoul(value(), nullptr, 10));
        } else if (arg == "--legacy") {
            o.legacy = true;
        } else if (arg == "--simulate-hang") {
            o.simulate_hang = true;
        } else if (arg == "--sweep") {
            o.sweep = true;
        } else if (arg == "--timed") {
            o.timed = true;
        } else if (arg == "--work") {
            o.work_us = std::atof(value());
        } else if (arg == "--replays") {
            o.replays = std::atoi(value());
        } else if (arg[0] != '-' && positional < 2) {
            o.device[positional++] = std::atoi(arg.c_str());
        } else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            std::exit(1);
        }
    }
    if (o.payload == 0 || o.payload % sizeof(PeerVec) != 0 || o.chain < 1 || o.replays < 1) {
        std::fprintf(stderr, "payload must be a positive multiple of 16 bytes, chain and replays >= 1\n");
        std::exit(1);
    }
    return o;
}

// Deterministic BF16 operands: rank r, exchange k, element i.
float operand(int rank, int k, int i) {
    const int v = (k * 7 + i * 13 + rank * 101) % 257 - 128;
    return static_cast<float>(v) / 16.0F;
}

constexpr int kSlots = 2; // the engine's slot count: exchanges alternate between two slots

// One pinned slab, mapped into both devices, laid out as PeerMailbox lays it out: [rank][slot]
// payloads, then [rank][slot][lane] 64-byte flag lines (lanes for one vector per lane, the most
// any variant uses), then the hang word's line.
struct Slab {
    void* host             = nullptr;
    std::size_t slot_bytes = 0;
    std::size_t lanes      = 0;
    std::size_t flag_words = 0; // per rank
    std::size_t payload_bytes = 0;
    std::size_t bytes      = 0;
    PeerVecBf16* payload_dev[2] = {nullptr, nullptr}; // device-side view per DEVICE (index = rank)
    std::uint32_t* flag_dev[2]  = {nullptr, nullptr};
    std::uint32_t* hang_dev[2]  = {nullptr, nullptr};
    std::uint32_t* flag_host    = nullptr;
    std::uint32_t* hang_host    = nullptr;
    PeerVecBf16* payload(int viewer, int rank, int slot) const {
        return payload_dev[viewer] + (static_cast<std::size_t>(rank) * kSlots + slot) * (slot_bytes / sizeof(PeerVec));
    }
    std::uint32_t* flags(int viewer, int rank, int slot) const {
        return flag_dev[viewer] + static_cast<std::size_t>(rank) * flag_words +
               static_cast<std::size_t>(slot) * lanes * pd::kPeerFlagStride;
    }
};

Slab alloc_slab(const Options& o, std::size_t payload) {
    Slab s;
    s.slot_bytes    = (payload + 255) / 256 * 256;
    s.lanes         = static_cast<std::size_t>(pd::peer_pipelined_warps(s.slot_bytes / sizeof(PeerVec), 1));
    s.flag_words    = kSlots * s.lanes * pd::kPeerFlagStride;
    s.payload_bytes = 2 * kSlots * s.slot_bytes;
    s.bytes         = s.payload_bytes + (2 * s.flag_words + pd::kPeerFlagStride) * sizeof(std::uint32_t);
    CK(cudaHostAlloc(&s.host, s.bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(s.host, 0, s.bytes);
    s.flag_host = reinterpret_cast<std::uint32_t*>(static_cast<char*>(s.host) + s.payload_bytes);
    s.hang_host = s.flag_host + 2 * s.flag_words;
    for (int rank = 0; rank < 2; ++rank) {
        CK(cudaSetDevice(o.device[rank]));
        void* base = nullptr;
        CK(cudaHostGetDevicePointer(&base, s.host, 0));
        s.payload_dev[rank] = static_cast<PeerVecBf16*>(base);
        s.flag_dev[rank]    = reinterpret_cast<std::uint32_t*>(static_cast<char*>(base) + s.payload_bytes);
        s.hang_dev[rank]    = s.flag_dev[rank] + 2 * s.flag_words;
    }
    return s;
}

struct Rank {
    int device              = 0;
    cudaStream_t stream     = nullptr;
    PeerVecBf16* partial    = nullptr; // [chain][vecs]
    PeerVecBf16* staged     = nullptr; // copies path scratch, [chain][vecs]
    std::uint32_t* words    = nullptr; // epochs (and the original kernel's arrival counters)
    std::size_t word_count  = 0;
    unsigned long long* stamps = nullptr; // [chain][stamp stride]
    std::vector<std::uint16_t> host;    // host copy of partial for init / verification
};

struct Probe {
    Options o;
    Slab slab;
    Rank rank[2];
    int vecs  = 0;
    int elems = 0;
    int stamp_stride = 0;

    void setup(const Options& opts, std::size_t payload, int chain) {
        o         = opts;
        o.payload = payload;
        o.chain   = chain;
        vecs      = static_cast<int>(payload / sizeof(PeerVec));
        elems     = vecs * 8;
        slab      = alloc_slab(o, payload);
        stamp_stride = std::max<int>(kLegacyStamps + 1, static_cast<int>(slab.lanes) * pd::kPeerStampCount);
        for (int r = 0; r < 2; ++r) {
            Rank& rk  = rank[r];
            rk.device = o.device[r];
            CK(cudaSetDevice(rk.device));
            CK(cudaStreamCreateWithFlags(&rk.stream, cudaStreamNonBlocking));
            const std::size_t bytes = static_cast<std::size_t>(chain) * payload;
            CK(cudaMalloc(&rk.partial, bytes));
            CK(cudaMalloc(&rk.staged, bytes));
            rk.word_count = std::max<std::size_t>(2 * kSlots, kSlots * slab.lanes);
            CK(cudaMalloc(&rk.words, rk.word_count * sizeof(std::uint32_t)));
            CK(cudaMalloc(&rk.stamps, static_cast<std::size_t>(chain) * stamp_stride * sizeof(unsigned long long)));
            rk.host.resize(static_cast<std::size_t>(chain) * elems);
        }
    }

    void init_partials() {
        for (int r = 0; r < 2; ++r) {
            for (int k = 0; k < o.chain; ++k) {
                for (int i = 0; i < elems; ++i) {
                    const __nv_bfloat16 b = __float2bfloat16_rn(operand(r, k, i));
                    std::memcpy(&rank[r].host[static_cast<std::size_t>(k) * elems + i], &b, 2);
                }
            }
            CK(cudaSetDevice(rank[r].device));
            CK(cudaMemcpy(rank[r].partial, rank[r].host.data(), rank[r].host.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemset(rank[r].words, 0, rank[r].word_count * sizeof(std::uint32_t)));
            CK(cudaMemset(rank[r].stamps, 0, static_cast<std::size_t>(o.chain) * stamp_stride * sizeof(unsigned long long)));
            CK(cudaDeviceSynchronize());
        }
        std::memset(slab.flag_host, 0, (2 * slab.flag_words + 1) * sizeof(std::uint32_t));
    }

    void launch_exchange(const Variant& v, int r, int k) {
        CK(cudaSetDevice(rank[r].device));
        const int slot       = k % kSlots;
        PeerVecBf16* partial = rank[r].partial + static_cast<std::size_t>(k) * vecs;
        PeerVecBf16* mine    = slab.payload(r, r, slot);
        const PeerVecBf16* peer = slab.payload(r, 1 - r, slot);
        std::uint32_t* mine_flags = slab.flags(r, r, slot);
        std::uint32_t* peer_flags = slab.flags(r, 1 - r, slot);
        unsigned long long* stamps = rank[r].stamps + static_cast<std::size_t>(k) * stamp_stride;
        const int blocks = v.blocks(static_cast<std::size_t>(vecs));
        if (v.kind == Kind::Legacy) {
            pd::peer_exchange_sum_kernel<<<blocks, 256, 0, rank[r].stream>>>(
                partial, mine, mine_flags, peer, peer_flags, rank[r].words + kSlots + slot,
                rank[r].words + slot, slab.hang_dev[r], vecs);
        } else if (v.kind == Kind::LegacyTimed) {
            legacy_timed_kernel<<<blocks, 256, 0, rank[r].stream>>>(
                partial, mine, mine_flags, peer, peer_flags, rank[r].words + kSlots + slot,
                rank[r].words + slot, slab.hang_dev[r], vecs, stamps);
        } else {
            pd::PeerPipelinedTuning tuning;
            tuning.spin_limit = o.spin_limit;
            tuning.sleep_ns   = v.sleep;
            tuning.stamps     = v.timed ? stamps : nullptr;
            std::uint32_t* epochs = rank[r].words + static_cast<std::size_t>(slot) * slab.lanes;
            switch (v.vecs_per_lane) {
            case 1: launch_pipelined_g<1>(v, blocks, rank[r].stream, partial, mine, mine_flags, peer, peer_flags, epochs, slab.hang_dev[r], vecs, tuning); break;
            case 2: launch_pipelined_g<2>(v, blocks, rank[r].stream, partial, mine, mine_flags, peer, peer_flags, epochs, slab.hang_dev[r], vecs, tuning); break;
            case 4: launch_pipelined_g<4>(v, blocks, rank[r].stream, partial, mine, mine_flags, peer, peer_flags, epochs, slab.hang_dev[r], vecs, tuning); break;
            case 8: launch_pipelined_g<8>(v, blocks, rank[r].stream, partial, mine, mine_flags, peer, peer_flags, epochs, slab.hang_dev[r], vecs, tuning); break;
            default: std::fprintf(stderr, "unsupported vecs per lane %d\n", v.vecs_per_lane); std::exit(1);
            }
        }
        CK(cudaGetLastError());
    }

    void sync_both() {
        for (int r = 0; r < 2; ++r) { CK(cudaSetDevice(rank[r].device)); CK(cudaDeviceSynchronize()); }
    }

    bool hang() const { return *static_cast<volatile std::uint32_t*>(slab.hang_host) != 0; }

    // Every exchange k of the chain must hold, bit for bit, the BF16 rounding of the FP32 sum of the
    // two BF16 operands, on both ranks; `launches` counts graph launches since init (each launch
    // runs every exchange once; after the first one each further launch doubles, exactly).
    int verify(const char* what, int exchanges) {
        int wrong = 0;
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(rank[r].device));
            std::vector<std::uint16_t> got(rank[r].host.size());
            CK(cudaMemcpy(got.data(), rank[r].partial, got.size() * 2, cudaMemcpyDeviceToHost));
            for (int k = 0; k < exchanges; ++k) {
                for (int i = 0; i < elems; ++i) {
                    const __nv_bfloat16 expect = __float2bfloat16_rn(
                        __bfloat162float(__float2bfloat16_rn(operand(0, k, i))) +
                        __bfloat162float(__float2bfloat16_rn(operand(1, k, i))));
                    std::uint16_t e = 0;
                    std::memcpy(&e, &expect, 2);
                    if (got[static_cast<std::size_t>(k) * elems + i] != e) { ++wrong; }
                }
            }
        }
        if (wrong != 0) {
            std::printf("  %s: WRONG SUMS (%d elements differ from the CPU reference)\n", what, wrong);
        }
        return wrong;
    }

    // Captures `count` exchanges of variant `v` on both ranks into one graph rooted on rank 0's
    // stream (an optional spin kernel before each on both ranks); with skip_rank_1 the peer never
    // publishes (hang report path).
    cudaGraphExec_t capture_mailbox_chain(const Variant& v, int count, bool skip_rank_1) {
        cudaEvent_t fork = nullptr, join = nullptr;
        CK(cudaSetDevice(rank[0].device)); CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        CK(cudaSetDevice(rank[1].device)); CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
        CK(cudaSetDevice(rank[0].device));
        cudaGraph_t graph = nullptr;
        CK(cudaStreamBeginCapture(rank[0].stream, cudaStreamCaptureModeThreadLocal));
        CK(cudaEventRecord(fork, rank[0].stream));
        CK(cudaSetDevice(rank[1].device));
        CK(cudaStreamWaitEvent(rank[1].stream, fork, 0));
        const auto work_ns = static_cast<unsigned long long>(o.work_us * 1000.0);
        for (int k = 0; k < count; ++k) {
            for (int r = 0; r < 2; ++r) {
                if (r == 1 && skip_rank_1) { continue; }
                if (work_ns > 0) {
                    CK(cudaSetDevice(rank[r].device));
                    spin_kernel<<<1, 32, 0, rank[r].stream>>>(work_ns);
                    CK(cudaGetLastError());
                }
                launch_exchange(v, r, k);
            }
        }
        CK(cudaSetDevice(rank[1].device));
        CK(cudaEventRecord(join, rank[1].stream));
        CK(cudaSetDevice(rank[0].device));
        CK(cudaStreamWaitEvent(rank[0].stream, join, 0));
        CK(cudaStreamEndCapture(rank[0].stream, &graph));
        cudaGraphExec_t exec = nullptr;
        CK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
        CK(cudaGraphDestroy(graph));
        CK(cudaEventDestroy(fork));
        CK(cudaEventDestroy(join));
        return exec;
    }

    // The copies path: for exchange k both ranks copy the peer's partial (cudaMemcpyAsync over UVA,
    // staged by the driver without peer access) after an event hop, then, after a second hop that
    // confirms the peer has finished reading this rank's partial, add in place.
    cudaGraphExec_t capture_copies_chain(int count) {
        std::vector<cudaEvent_t> ready(2 * count), copied(2 * count);
        for (int k = 0; k < count; ++k) {
            for (int r = 0; r < 2; ++r) {
                CK(cudaSetDevice(rank[r].device));
                CK(cudaEventCreateWithFlags(&ready[2 * k + r], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&copied[2 * k + r], cudaEventDisableTiming));
            }
        }
        cudaEvent_t fork = nullptr, join = nullptr;
        CK(cudaSetDevice(rank[0].device)); CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        CK(cudaSetDevice(rank[1].device)); CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
        CK(cudaSetDevice(rank[0].device));
        cudaGraph_t graph = nullptr;
        CK(cudaStreamBeginCapture(rank[0].stream, cudaStreamCaptureModeThreadLocal));
        CK(cudaEventRecord(fork, rank[0].stream));
        CK(cudaSetDevice(rank[1].device));
        CK(cudaStreamWaitEvent(rank[1].stream, fork, 0));
        for (int k = 0; k < count; ++k) {
            const std::size_t off = static_cast<std::size_t>(k) * vecs;
            for (int r = 0; r < 2; ++r) { CK(cudaSetDevice(rank[r].device)); CK(cudaEventRecord(ready[2 * k + r], rank[r].stream)); }
            for (int r = 0; r < 2; ++r) {
                CK(cudaSetDevice(rank[r].device));
                CK(cudaStreamWaitEvent(rank[r].stream, ready[2 * k + (1 - r)], 0));
                CK(cudaMemcpyAsync(rank[r].staged + off, rank[1 - r].partial + off, o.payload,
                                   cudaMemcpyDeviceToDevice, rank[r].stream));
                CK(cudaEventRecord(copied[2 * k + r], rank[r].stream));
            }
            for (int r = 0; r < 2; ++r) {
                CK(cudaSetDevice(rank[r].device));
                CK(cudaStreamWaitEvent(rank[r].stream, copied[2 * k + (1 - r)], 0));
                add_bf16_kernel<<<(vecs + 255) / 256, 256, 0, rank[r].stream>>>(rank[r].partial + off, rank[r].staged + off, vecs);
                CK(cudaGetLastError());
            }
        }
        CK(cudaSetDevice(rank[1].device));
        CK(cudaEventRecord(join, rank[1].stream));
        CK(cudaSetDevice(rank[0].device));
        CK(cudaStreamWaitEvent(rank[0].stream, join, 0));
        CK(cudaStreamEndCapture(rank[0].stream, &graph));
        cudaGraphExec_t exec = nullptr;
        CK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
        CK(cudaGraphDestroy(graph));
        return exec;
    }

    double launch_graph(cudaGraphExec_t exec) {
        CK(cudaSetDevice(rank[0].device));
        const auto start = std::chrono::steady_clock::now();
        CK(cudaGraphLaunch(exec, rank[0].stream));
        sync_both();
        return ms_since(start);
    }

    // GPU time of `replays` back-to-back launches, from events on rank 0's stream (the graph
    // joins rank 1 back into it), per launch, in ms.
    double replay_ms(cudaGraphExec_t exec, int replays) {
        CK(cudaSetDevice(rank[0].device));
        cudaEvent_t start = nullptr, stop = nullptr;
        CK(cudaEventCreate(&start));
        CK(cudaEventCreate(&stop));
        CK(cudaEventRecord(start, rank[0].stream));
        for (int i = 0; i < replays; ++i) { CK(cudaGraphLaunch(exec, rank[0].stream)); }
        CK(cudaEventRecord(stop, rank[0].stream));
        sync_both();
        float ms = 0.0F;
        CK(cudaEventElapsedTime(&ms, start, stop));
        CK(cudaEventDestroy(start));
        CK(cudaEventDestroy(stop));
        return static_cast<double>(ms) / replays;
    }

    // One variant's graph chain: correctness after the first launch, then the mean GPU time per
    // exchange over o.replays launches (the --work spin excluded). Returns -1 on a hang.
    double measure(const Variant& v, int& wrong) {
        init_partials();
        cudaGraphExec_t exec = capture_mailbox_chain(v, o.chain, false);
        launch_graph(exec); // first launch: every exchange k holds operand(0,k) + operand(1,k)
        if (hang()) { CK(cudaGraphExecDestroy(exec)); return -1.0; }
        wrong += verify(v.name().c_str(), o.chain);
        launch_graph(exec); // warm
        const double ms = replay_ms(exec, o.replays);
        CK(cudaGraphExecDestroy(exec));
        if (hang()) { return -1.0; }
        return ms * 1000.0 / o.chain - o.work_us;
    }

    std::vector<unsigned long long> stamps(int r) {
        CK(cudaSetDevice(rank[r].device));
        std::vector<unsigned long long> out(static_cast<std::size_t>(o.chain) * stamp_stride);
        CK(cudaMemcpy(out.data(), rank[r].stamps, out.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        return out;
    }
};

void print_environment(const Options& o) {
    int driver = 0, runtime = 0;
    CK(cudaDriverGetVersion(&driver));
    CK(cudaRuntimeGetVersion(&runtime));
    std::printf("CUDA driver %d.%d, runtime %d.%d\n", driver / 1000, driver % 1000 / 10, runtime / 1000, runtime % 1000 / 10);
#ifdef _WIN32
    std::printf("OS: Windows\n");
#else
    std::ifstream version("/proc/version");
    std::string line;
    std::getline(version, line);
    const bool wsl = line.find("microsoft") != std::string::npos || line.find("Microsoft") != std::string::npos;
    std::printf("OS: Linux%s\n", wsl ? " under WSL2 (Microsoft kernel)" : "");
#endif
    for (int r = 0; r < 2; ++r) {
        cudaDeviceProp p{};
        CK(cudaGetDeviceProperties(&p, o.device[r]));
        std::printf("device %d: %s, sm_%d%d, PCI %04x:%02x:%02x, %.1f GiB, mapHost %d, UVA %d, hostNativeAtomics %d\n",
                    o.device[r], p.name, p.major, p.minor, p.pciDomainID, p.pciBusID, p.pciDeviceID,
                    static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                    p.canMapHostMemory, p.unifiedAddressing, p.hostNativeAtomicSupported);
    }
    int ab = 0, ba = 0;
    CK(cudaDeviceCanAccessPeer(&ab, o.device[0], o.device[1]));
    CK(cudaDeviceCanAccessPeer(&ba, o.device[1], o.device[0]));
    std::printf("peer access %d->%d: %d, %d->%d: %d (0 = no P2P; both transports work without it)\n",
                o.device[0], o.device[1], ab, o.device[1], o.device[0], ba);
}

double median(std::vector<double> v) {
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// --sweep: every variant at 4, 10 and 40 KiB (and --payload if different).
int run_sweep(const Options& o) {
    std::vector<std::size_t> payloads = {4096, 10240, 40960};
    if (std::find(payloads.begin(), payloads.end(), o.payload) == payloads.end()) { payloads.push_back(o.payload); }
    std::vector<Variant> variants;
    variants.push_back(Variant{Kind::Legacy});
    for (int g : {1, 2, 4, 8}) {
        for (int t : {64, 128, 256}) {
            for (int poll : {pd::kPeerPollAcquire, pd::kPeerPollEarlyFence}) {
                variants.push_back(Variant{Kind::Pipelined, g, t, poll, pd::kPeerPollSleepNs, false});
            }
        }
    }
    for (std::uint32_t sleep : {0u, 32u}) {
        variants.push_back(Variant{Kind::Pipelined, pd::kPeerPipelinedVecsPerLane, pd::kPeerPipelinedThreads,
                                   pd::kPeerPollAcquire, sleep, false});
    }
    int wrong = 0;
    std::printf("\n[sweep] us per exchange, graph chain x%d, 2 alternating slots, mean of %d replays%s\n", o.chain,
                o.replays, o.work_us > 0 ? " (spin work excluded)" : "");
    std::printf("%-8s %-42s %6s %10s\n", "payload", "kernel", "blocks", "us/exch");
    for (const std::size_t payload : payloads) {
        Probe p;
        p.setup(o, payload, o.chain);
        for (const Variant& v : variants) {
            const double us = p.measure(v, wrong);
            std::printf("%-8zu %-42s %6d %10.2f%s\n", payload, v.name().c_str(), v.blocks(payload / 16), us,
                        us < 0 ? "  HANG" : "");
            if (us < 0) { return 1; }
        }
    }
    if (wrong != 0) { std::printf("sweep: WRONG SUMS in %d elements\n", wrong); return 1; }
    std::printf("sweep: every variant bit-exact against the CPU reference\n");
    return 0;
}

// --timed: per-phase anatomy of both kernels at --payload. Stamps are clock64 on the SM that ran
// warp 0 / block 0, converted with that SM's clock measured against globaltimer over the chain.
int run_timed(const Options& o) {
    Probe p;
    p.setup(o, o.payload, o.chain);
    int wrong = 0;
    std::printf("\n[timed] anatomy at %zu bytes, chain x%d%s: medians over exchanges, us\n", o.payload, o.chain,
                o.work_us > 0 ? " with spin work between exchanges" : "");
    struct Row { const char* label; double early; double late; };
    for (int which = 0; which < 2; ++which) {
        Variant v = which == 0 ? Variant{Kind::LegacyTimed}
                               : Variant{Kind::Pipelined, pd::kPeerPipelinedVecsPerLane, pd::kPeerPipelinedThreads,
                                         pd::kPeerPollAcquire, pd::kPeerPollSleepNs, true};
        p.init_partials();
        cudaGraphExec_t exec = p.capture_mailbox_chain(v, o.chain, false);
        p.launch_graph(exec);
        if (p.hang()) { std::printf("  HANG\n"); return 1; }
        wrong += p.verify(v.name().c_str(), o.chain);
        CK(cudaGraphExecDestroy(exec));
        const int warps = pd::peer_pipelined_warps(static_cast<std::size_t>(p.vecs), pd::kPeerPipelinedVecsPerLane);
        std::printf("  %s (%d blocks%s)\n", v.name().c_str(), v.blocks(static_cast<std::size_t>(p.vecs)),
                    which == 0 ? ", thread 0 of block 0" : (", warp 0 and the slowest of " + std::to_string(warps) + " warps").c_str());
        for (int r = 0; r < 2; ++r) {
            const auto st = p.stamps(r);
            double cycles = 0, ns = 0;
            std::vector<double> ph[8];
            std::vector<double> total, slowest_end, slowest_seen;
            int immediate = 0;
            for (int k = 0; k < o.chain; ++k) {
                const unsigned long long* s = st.data() + static_cast<std::size_t>(k) * p.stamp_stride;
                if (which == 0) {
                    cycles += static_cast<double>(s[8] - s[1]);
                    ns += static_cast<double>(s[9] - s[0]);
                } else {
                    cycles += static_cast<double>(s[5] - s[1]);
                    ns += static_cast<double>(s[6] - s[0]);
                }
            }
            const double ghz = cycles / ns; // cycles per ns
            for (int k = 0; k < o.chain; ++k) {
                const unsigned long long* s = st.data() + static_cast<std::size_t>(k) * p.stamp_stride;
                auto us = [&](unsigned long long a, unsigned long long b) { return static_cast<double>(b - a) / ghz / 1000.0; };
                if (which == 0) {
                    ph[0].push_back(us(s[1], s[2])); // publish loop (4 dependent load->store pairs)
                    ph[1].push_back(us(s[2], s[3])); // __threadfence_system
                    ph[2].push_back(us(s[3], s[4])); // __syncthreads + arrival atomic (+ release)
                    ph[3].push_back(us(s[4], s[5])); // poll until the peer's flag
                    ph[4].push_back(us(s[5], s[6])); // acquire fence
                    ph[5].push_back(us(s[6], s[7])); // __syncthreads
                    ph[6].push_back(us(s[7], s[8])); // read loop + combine (4 dependent round trips)
                    total.push_back(us(s[1], s[8]));
                    if (s[kLegacyStamps] == 0) { ++immediate; }
                } else {
                    ph[0].push_back(us(s[1], s[2])); // loads + stores issued
                    ph[1].push_back(us(s[2], s[3])); // release (system fence + flag store)
                    ph[2].push_back(us(s[3], s[4])); // poll until the peer's flag (acquire)
                    ph[3].push_back(us(s[4], s[5])); // peer loads + combine
                    total.push_back(us(s[1], s[5]));
                    if (s[7] == 0) { ++immediate; }
                    double worst_end = 0, worst_seen = 0;
                    for (int w = 0; w < warps; ++w) {
                        const unsigned long long* sw = s + static_cast<std::size_t>(w) * pd::kPeerStampCount;
                        // globaltimer is shared by the device's SMs: warp w's end against warp 0's entry
                        worst_end  = std::max(worst_end, static_cast<double>(sw[6] - s[0]) / 1000.0);
                        worst_seen = std::max(worst_seen, us(sw[1], sw[4]));
                    }
                    slowest_end.push_back(worst_end);
                    slowest_seen.push_back(worst_seen);
                }
            }
            std::printf("    rank %d (SM clock %.2f GHz, peer flag already there at the first poll in %d/%d):\n", r, ghz,
                        immediate, o.chain);
            if (which == 0) {
                std::printf("      publish %.2f | fence.sc.sys %.2f | bar+arrival+release %.2f | poll %.2f | acquire fence %.2f | bar %.2f | read+combine %.2f | total %.2f\n",
                            median(ph[0]), median(ph[1]), median(ph[2]), median(ph[3]), median(ph[4]), median(ph[5]),
                            median(ph[6]), median(total));
            } else {
                std::printf("      loads+stores %.2f | release %.2f | poll %.2f | read+combine %.2f | total %.2f | slowest warp: flag seen %.2f, end %.2f (globaltimer)\n",
                            median(ph[0]), median(ph[1]), median(ph[2]), median(ph[3]), median(total),
                            median(slowest_seen), median(slowest_end));
            }
        }
    }
    if (wrong != 0) { std::printf("timed: WRONG SUMS in %d elements\n", wrong); return 1; }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0); // a hang must be visible up to the line it happened at
    const Options o = parse(argc, argv);
    int devices     = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::printf("skip: fewer than two CUDA devices\n");
        return 77;
    }
    if (o.device[0] == o.device[1] || o.device[0] >= devices || o.device[1] >= devices) {
        std::fprintf(stderr, "device ids must be two distinct ids below %d\n", devices);
        return 1;
    }
    const Variant chosen = o.legacy ? Variant{Kind::Legacy} : Variant{};
    std::printf("NInfer TP2 mailbox probe: payload %zu bytes, chain %d, %s kernel, spin limit %u%s\n", o.payload,
                o.chain, o.legacy ? "legacy" : "pipelined", o.spin_limit, o.simulate_hang ? ", SIMULATED HANG" : "");
    print_environment(o);
    int wrong = 0;

    // ---- 1. the startup probe: one 4 KiB exchange captured in a two-device graph ----------------
    std::printf("\n[1] startup probe: one exchange captured in a two-device graph (engine rule: hang or > 50 ms => copies)\n");
    Probe s;
    s.setup(o, 4096, 1);
    s.init_partials();
    cudaGraphExec_t startup = s.capture_mailbox_chain(chosen, 1, o.simulate_hang);
    const double startup_ms = s.launch_graph(startup);
    const bool startup_hang = s.hang();
    bool mailbox_ok         = !startup_hang && startup_ms <= 50.0;
    if (!startup_hang) { wrong += s.verify("startup exchange", 1); }
    std::printf("  exchange: %.3f ms, hang word %u => %s\n", startup_ms, startup_hang ? 1u : 0u,
                startup_hang ? "TIMED OUT (the peer's flag never arrived; the engine would fall back to copies)"
                : mailbox_ok ? "mailbox usable" : "too slow (the engine would fall back to copies)");
    CK(cudaGraphExecDestroy(startup));
    if (o.simulate_hang) {
        std::printf("\nsimulated hang: %s\n", startup_hang ? "detected as expected" : "NOT DETECTED");
        return startup_hang ? 2 : 1;
    }
    if (mailbox_ok && o.sweep) { return run_sweep(o) != 0 || wrong != 0 ? 1 : 0; }
    if (mailbox_ok && o.timed) { return run_timed(o) != 0 || wrong != 0 ? 1 : 0; }

    Probe p;
    p.setup(o, o.payload, o.chain);
    if (mailbox_ok) {
        // ---- 2. mailbox timings at the decode payload ------------------------------------------
        std::printf("\n[2] mailbox at %zu bytes, graph chain x%d, 2 alternating slots\n", o.payload, o.chain);
        const Variant other = o.legacy ? Variant{} : Variant{Kind::Legacy};
        for (const Variant& v : {chosen, other}) {
            const double us = p.measure(v, wrong);
            if (us < 0) {
                std::printf("  HANG in the captured chain (%s kernel)\n", v.name().c_str());
                mailbox_ok = false;
                break;
            }
            std::printf("  %s kernel: %.1f us per exchange (mean of %d replays)\n", v.name().c_str(), us, o.replays);
        }
    } else {
        std::printf("\n[2] mailbox timings skipped (mailbox unusable)\n");
    }

    // ---- 3. the copies path, for comparison and as the fallback's health check -----------------
    std::printf("\n[3] copies path at %zu bytes (cudaMemcpyAsync over UVA between two event hops, in a graph)\n", o.payload);
    bool copies_ok = true;
    {
        p.init_partials();
        cudaGraphExec_t exec = p.capture_copies_chain(o.chain);
        const double first   = p.launch_graph(exec);
        const int bad        = p.verify("copies chain, first replay", o.chain);
        wrong += bad;
        copies_ok = bad == 0;
        const double ms = p.replay_ms(exec, o.replays);
        std::printf("  graph chain x%d: first replay %.3f ms, then %.3f ms per replay = %.1f us per exchange\n",
                    o.chain, first, ms, ms * 1000.0 / o.chain);
        CK(cudaGraphExecDestroy(exec));
    }

    std::printf("\nVERDICT: mailbox %s, copies %s%s\n", mailbox_ok ? "USABLE" : "UNUSABLE", copies_ok ? "ok" : "BROKEN",
                mailbox_ok ? " (the engine's default transport works here)"
                           : " (the engine falls back to copies at startup; on builds before the probe pass --no-tp-mailbox)");
    if (wrong != 0) { std::printf("wrong sums: %d elements\n", wrong); return 1; }
    return mailbox_ok ? 0 : (copies_ok ? 2 : 1);
}
