// Standalone TP2 mailbox probe: does the pinned-host mailbox exchange work between these two GPUs?
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
// The exchange kernel below is the production kernel (src/ops/kernel/peer_exchange.cuh) with the
// spin limit as a parameter; the mailbox is allocated as PeerMailbox does (pinned, mapped,
// portable). No engine headers: nvcc and the CUDA runtime only.
//
// Build:
//   Linux    nvcc -O2 -std=c++20 -arch=sm_120a -o mailbox_probe tools/tp2/mailbox_probe.cu
//   Windows  nvcc -O2 -std=c++20 -arch=sm_120a -o mailbox_probe.exe tools\tp2\mailbox_probe.cu
//   (-arch=native also works on CUDA 12.8+; the exchange needs no sm_120-specific feature.)
// Run:
//   mailbox_probe [dev_a dev_b] [--payload BYTES] [--chain N] [--spin-limit N] [--simulate-hang]
//     dev_a dev_b     CUDA device ids (default 0 1)
//     --payload       exchange payload in bytes, multiple of 16 (default 10240, the decode activation)
//     --chain         exchanges per chain (default 128, one decode step's worth)
//     --spin-limit    poller probes before an exchange reports a hang (default 1000000, ~0.8 s)
//     --simulate-hang rank 1 skips the startup exchange, to exercise the hang report path
// Exit status: 0 mailbox usable · 2 mailbox unusable (timed out or too slow) while the copies path
// works · 1 CUDA error or wrong sums · 77 fewer than two devices.
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

// ---- the production exchange kernel (peer_exchange.cuh), spin limit as a parameter -----------
constexpr std::uint32_t kHangProbe = 1024u;
constexpr int kGroup               = 4;
using PeerVec                      = uint4;
union PeerVecBf16 {
    PeerVec raw;
    __nv_bfloat162 pair[4];
};

__global__ __launch_bounds__(256) void peer_exchange_sum_kernel(
    PeerVecBf16* partial, PeerVecBf16* mine_payload, std::uint32_t* mine_flag,
    const PeerVecBf16* peer_payload, std::uint32_t* peer_flag, std::uint32_t* epoch,
    std::uint32_t* arrival, std::uint32_t* hang, int vecs, std::uint32_t spin_limit) {
    using DeviceWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>;
    using SystemWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;
    __shared__ int combine;
    const int tid        = blockIdx.x * blockDim.x + threadIdx.x;
    const int group_span = gridDim.x * blockDim.x * kGroup;
    std::uint32_t target = 0;
    if (threadIdx.x == 0) { target = DeviceWord(*epoch).load(cuda::memory_order_relaxed) + 1u; }
    for (int g = tid * kGroup; g < vecs; g += group_span) {
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
            if (g + j < vecs) { mine_payload[g + j].raw = partial[g + j].raw; }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
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
            if (spins % kHangProbe == 0u &&
                (spins > spin_limit || fault.load(cuda::memory_order_relaxed) != 0u)) {
                fault.store(1u, cuda::memory_order_relaxed);
                published = false;
                break;
            }
        }
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
        combine = published ? 1 : 0;
    }
    __syncthreads();
    if (combine == 0) { return; }
    for (int g = tid * kGroup; g < vecs; g += group_span) {
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
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

int exchange_blocks(std::size_t bytes) {
    const std::size_t vecs = bytes / sizeof(PeerVec);
    const std::size_t want = (vecs + kGroup * 256 - 1) / (kGroup * 256);
    return want < 1 ? 1 : (want > 16 ? 16 : static_cast<int>(want));
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

// ---- host side ---------------------------------------------------------------------------------
double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct Options {
    int device[2]            = {0, 1};
    std::size_t payload      = 10240;
    int chain                = 128;
    std::uint32_t spin_limit = 1000000u;
    bool simulate_hang       = false;
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
        } else if (arg == "--simulate-hang") {
            o.simulate_hang = true;
        } else if (arg[0] != '-' && positional < 2) {
            o.device[positional++] = std::atoi(arg.c_str());
        } else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            std::exit(1);
        }
    }
    if (o.payload == 0 || o.payload % sizeof(PeerVec) != 0 || o.chain < 1) {
        std::fprintf(stderr, "payload must be a positive multiple of 16 bytes, chain >= 1\n");
        std::exit(1);
    }
    return o;
}

// Deterministic BF16 operands: rank r, exchange k, element i.
float operand(int rank, int k, int i) {
    const int v = (k * 7 + i * 13 + rank * 101) % 257 - 128;
    return static_cast<float>(v) / 16.0F;
}

// One pinned slab, mapped into both devices: [rank][slot] payloads, then [rank][slot] flags, then
// the hang word.
struct Slab {
    void* host                  = nullptr;
    std::size_t slots           = 0;
    std::size_t slot_bytes      = 0;
    std::size_t payload_bytes   = 0;
    PeerVecBf16* payload_dev[2] = {nullptr, nullptr}; // device-side view per DEVICE (index = rank)
    std::uint32_t* flag_dev[2]  = {nullptr, nullptr};
    std::uint32_t* hang_dev[2]  = {nullptr, nullptr};
    std::uint32_t* flag_host    = nullptr;
    std::uint32_t* hang_host    = nullptr;
    PeerVecBf16* payload(int viewer, int rank, std::size_t slot) const {
        return payload_dev[viewer] + (static_cast<std::size_t>(rank) * slots + slot) * (slot_bytes / sizeof(PeerVec));
    }
    std::uint32_t* flag(int viewer, int rank, std::size_t slot) const {
        return flag_dev[viewer] + static_cast<std::size_t>(rank) * slots + slot;
    }
};

Slab alloc_slab(const Options& o, std::size_t slots) {
    Slab s;
    s.slots         = slots;
    s.slot_bytes    = (o.payload + 255) / 256 * 256;
    s.payload_bytes = 2 * slots * s.slot_bytes;
    const std::size_t words_bytes = (2 * slots + 1) * sizeof(std::uint32_t);
    const std::size_t bytes       = s.payload_bytes + (words_bytes + 255) / 256 * 256;
    CK(cudaHostAlloc(&s.host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(s.host, 0, bytes);
    s.flag_host = reinterpret_cast<std::uint32_t*>(static_cast<char*>(s.host) + s.payload_bytes);
    s.hang_host = s.flag_host + 2 * slots;
    for (int rank = 0; rank < 2; ++rank) {
        CK(cudaSetDevice(o.device[rank]));
        void* base = nullptr;
        CK(cudaHostGetDevicePointer(&base, s.host, 0));
        s.payload_dev[rank] = static_cast<PeerVecBf16*>(base);
        s.flag_dev[rank]    = reinterpret_cast<std::uint32_t*>(static_cast<char*>(base) + s.payload_bytes);
        s.hang_dev[rank]    = s.flag_dev[rank] + 2 * slots;
    }
    return s;
}

struct Rank {
    int device                 = 0;
    cudaStream_t stream        = nullptr;
    PeerVecBf16* partial       = nullptr; // [chain][vecs]
    PeerVecBf16* staged        = nullptr; // copies path scratch, [chain][vecs]
    std::uint32_t* epoch       = nullptr; // [slots]
    std::uint32_t* arrival     = nullptr; // [slots]
    std::vector<std::uint16_t> host;      // host copy of partial for init / verification
};

struct Probe {
    Options o;
    Slab slab;
    Rank rank[2];
    int vecs = 0;
    int elems = 0;

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
            CK(cudaMemset(rank[r].epoch, 0, slab.slots * sizeof(std::uint32_t)));
            CK(cudaMemset(rank[r].arrival, 0, slab.slots * sizeof(std::uint32_t)));
            CK(cudaDeviceSynchronize());
        }
        std::memset(slab.flag_host, 0, (2 * slab.slots + 1) * sizeof(std::uint32_t));
    }

    void launch_exchange(int r, int k, std::size_t slot) {
        CK(cudaSetDevice(rank[r].device));
        PeerVecBf16* partial = rank[r].partial + static_cast<std::size_t>(k) * vecs;
        peer_exchange_sum_kernel<<<exchange_blocks(o.payload), 256, 0, rank[r].stream>>>(
            partial, slab.payload(r, r, slot), slab.flag(r, r, slot), slab.payload(r, 1 - r, slot),
            slab.flag(r, 1 - r, slot), rank[r].epoch + slot, rank[r].arrival + slot, slab.hang_dev[r],
            vecs, o.spin_limit);
        CK(cudaGetLastError());
    }

    void sync_both() {
        for (int r = 0; r < 2; ++r) { CK(cudaSetDevice(rank[r].device)); CK(cudaDeviceSynchronize()); }
    }

    bool hang() const { return *static_cast<volatile std::uint32_t*>(slab.hang_host) != 0; }

    // Every exchange k of the chain must hold operand(0,k,i) + operand(1,k,i), BF16-rounded, on both ranks.
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

    // Captures `count` exchanges (slot = exchange index) on both ranks into one graph rooted on rank 0's
    // stream; with skip_rank_1 the peer never publishes (hang report path).
    cudaGraphExec_t capture_mailbox_chain(int count, bool skip_rank_1) {
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
            launch_exchange(0, k, static_cast<std::size_t>(k));
            if (!skip_rank_1) { launch_exchange(1, k, static_cast<std::size_t>(k)); }
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
    std::printf("NInfer TP2 mailbox probe: payload %zu bytes, chain %d, spin limit %u%s\n", o.payload, o.chain,
                o.spin_limit, o.simulate_hang ? ", SIMULATED HANG" : "");
    print_environment(o);

    Probe p;
    p.o     = o;
    p.vecs  = static_cast<int>(o.payload / sizeof(PeerVec));
    p.elems = p.vecs * 8;
    p.slab  = alloc_slab(o, static_cast<std::size_t>(o.chain));
    for (int r = 0; r < 2; ++r) {
        Rank& rank  = p.rank[r];
        rank.device = o.device[r];
        CK(cudaSetDevice(rank.device));
        CK(cudaStreamCreateWithFlags(&rank.stream, cudaStreamNonBlocking));
        const std::size_t bytes = static_cast<std::size_t>(o.chain) * o.payload;
        CK(cudaMalloc(&rank.partial, bytes));
        CK(cudaMalloc(&rank.staged, bytes));
        CK(cudaMalloc(&rank.epoch, p.slab.slots * sizeof(std::uint32_t)));
        CK(cudaMalloc(&rank.arrival, p.slab.slots * sizeof(std::uint32_t)));
        rank.host.resize(static_cast<std::size_t>(o.chain) * p.elems);
    }
    int wrong = 0;

    // ---- 1. the startup probe: one 4 KiB exchange captured in a two-device graph ----------------
    std::printf("\n[1] startup probe: one exchange captured in a two-device graph (engine rule: hang or > 50 ms => copies)\n");
    Options small = o;
    small.payload = 4096;
    small.chain   = 1;
    Probe s;
    s.o     = small;
    s.vecs  = 256;
    s.elems = 2048;
    s.slab  = alloc_slab(small, 1);
    for (int r = 0; r < 2; ++r) {
        s.rank[r].device = o.device[r];
        s.rank[r].stream = p.rank[r].stream;
        CK(cudaSetDevice(s.rank[r].device));
        CK(cudaMalloc(&s.rank[r].partial, 4096));
        CK(cudaMalloc(&s.rank[r].staged, 4096));
        CK(cudaMalloc(&s.rank[r].epoch, sizeof(std::uint32_t)));
        CK(cudaMalloc(&s.rank[r].arrival, sizeof(std::uint32_t)));
        s.rank[r].host.resize(2048);
    }
    s.init_partials();
    cudaGraphExec_t startup = s.capture_mailbox_chain(1, o.simulate_hang);
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

    if (mailbox_ok) {
        // ---- 2. mailbox timings at the decode payload ------------------------------------------
        std::printf("\n[2] mailbox at %zu bytes\n", o.payload);
        p.init_partials();
        {
            const auto start = std::chrono::steady_clock::now();
            p.launch_exchange(0, 0, 0);
            p.launch_exchange(1, 0, 0);
            p.sync_both();
            std::printf("  one eager exchange pair (launch -> both devices idle): %.1f us\n", ms_since(start) * 1000.0);
            if (p.hang()) { std::printf("  HANG in the eager exchange\n"); mailbox_ok = false; }
        }
        if (mailbox_ok) {
            p.init_partials();
            const auto start = std::chrono::steady_clock::now();
            for (int k = 0; k < o.chain; ++k) { p.launch_exchange(0, k, k); p.launch_exchange(1, k, k); }
            p.sync_both();
            const double ms = ms_since(start);
            std::printf("  eager chain x%d: %.3f ms, %.1f us per exchange\n", o.chain, ms, ms * 1000.0 / o.chain);
            if (p.hang()) { std::printf("  HANG in the eager chain\n"); mailbox_ok = false; }
            else { wrong += p.verify("eager chain", o.chain); }
        }
        if (mailbox_ok) {
            p.init_partials();
            cudaGraphExec_t exec = p.capture_mailbox_chain(o.chain, false);
            const double first   = p.launch_graph(exec);
            if (p.hang()) { std::printf("  HANG in the captured chain\n"); mailbox_ok = false; }
            else {
                wrong += p.verify("graph chain, first replay", o.chain);
                double total = 0;
                for (int i = 0; i < 20; ++i) { total += p.launch_graph(exec); }
                std::printf("  graph chain x%d: first replay %.3f ms, then %.3f ms per replay = %.1f us per exchange\n",
                            o.chain, first, total / 20.0, total / 20.0 * 1000.0 / o.chain);
                if (p.hang()) { std::printf("  HANG in a graph replay\n"); mailbox_ok = false; }
            }
            CK(cudaGraphExecDestroy(exec));
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
        double total = 0;
        for (int i = 0; i < 20; ++i) { total += p.launch_graph(exec); }
        std::printf("  graph chain x%d: first replay %.3f ms, then %.3f ms per replay = %.1f us per exchange\n",
                    o.chain, first, total / 20.0, total / 20.0 * 1000.0 / o.chain);
        CK(cudaGraphExecDestroy(exec));
    }

    std::printf("\nVERDICT: mailbox %s, copies %s%s\n", mailbox_ok ? "USABLE" : "UNUSABLE", copies_ok ? "ok" : "BROKEN",
                mailbox_ok ? " (the engine's default transport works here)"
                           : " (the engine falls back to copies at startup; on builds before the probe pass --no-tp-mailbox)");
    if (wrong != 0) { std::printf("wrong sums: %d elements\n", wrong); return 1; }
    return mailbox_ok ? 0 : (copies_ok ? 2 : 1);
}
