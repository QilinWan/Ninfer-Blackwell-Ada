// Multi-device capture constraints (CUDA 13.1). Re-qualify them on any new two-GPU topology.
//
//  1. Two live captures cannot be linked: cudaStreamWaitEvent on an event recorded inside a
//     different live capture fails with cudaErrorStreamCaptureMerge. A tensor-parallel decode
//     program is therefore ONE graph holding both devices' nodes, with the peer stream forked into
//     the origin's capture (DecodeGraphPeerBridge).
//  2. Memcpy nodes must name their memory by UVA pointer. cudaMemcpyPeerAsync is rejected inside a
//     capture with cudaErrorStreamCaptureUnsupported; cudaMemcpyAsync(cudaMemcpyDeviceToDevice)
//     over UVA pointers is captured (see pull_peer() in src/ops/common/allreduce.cu).
//  3. cudaGraphExecUpdate requires the same topology, including node device residency. Profile
//     swaps work because every profile of a family is captured from the same body against the
//     same two DeviceContexts.
//  4. No device-side launch: device-launchable graphs must be single-device, so instantiation
//     keeps flags 0.
//  5. Event records and waits become edges, not nodes, so node_count() measures real device work
//     on both devices.

#include "core/decode_graph.h"

#include "core/device.h"
#include "core/device_scope.h"
#include "core/nvtx.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

// Fork the peer stream into the capture that `stream` (the origin) has already begun.
void fork_peer(cudaStream_t stream, const DecodeGraphPeerCapture& peer) {
    const ScopedCurrentDevice scope;
    ScopedCurrentDevice::select(peer.bridge->origin_device());
    CUDA_CHECK(cudaEventRecord(peer.bridge->fork_event(), stream));
    ScopedCurrentDevice::select(peer.bridge->peer_device());
    CUDA_CHECK(cudaStreamWaitEvent(peer.stream, peer.bridge->fork_event(), 0));
}

// Join the peer stream back into the origin. Without this cudaStreamEndCapture reports
// cudaErrorStreamCaptureUnjoined and the whole capture is discarded.
void join_peer(cudaStream_t stream, const DecodeGraphPeerCapture& peer) {
    const ScopedCurrentDevice scope;
    ScopedCurrentDevice::select(peer.bridge->peer_device());
    CUDA_CHECK(cudaEventRecord(peer.bridge->join_event(), peer.stream));
    ScopedCurrentDevice::select(peer.bridge->origin_device());
    CUDA_CHECK(cudaStreamWaitEvent(stream, peer.bridge->join_event(), 0));
}

void discard_capture(cudaStream_t stream, const DecodeGraphPeerCapture* peer,
                     bool peer_forked) noexcept {
    // Best effort: rejoin the peer so the origin's EndCapture is well formed. If the capture was
    // already invalidated these calls fail harmlessly and EndCapture then returns a null graph and
    // clears BOTH streams' capture state, which is the outcome that matters. Hand-rolled rather
    // than ScopedCurrentDevice because this runs on an exception path and must not throw.
    int caller_device = 0;
    const bool restore =
        peer != nullptr && peer_forked && cudaGetDevice(&caller_device) == cudaSuccess;
    if (peer != nullptr && peer_forked) {
        log_cuda_error("cudaSetDevice(peer)", cudaSetDevice(peer->bridge->peer_device()));
        log_cuda_error("cudaEventRecord(join)",
                       cudaEventRecord(peer->bridge->join_event(), peer->stream));
        log_cuda_error("cudaSetDevice(origin)", cudaSetDevice(peer->bridge->origin_device()));
        log_cuda_error("cudaStreamWaitEvent(join)",
                       cudaStreamWaitEvent(stream, peer->bridge->join_event(), 0));
    }
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
    if (peer != nullptr && peer_forked) {
        // A stream left in capture mode would poison every later launch on it, so say so loudly
        // rather than failing mysteriously later.
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(peer->stream, &status) == cudaSuccess &&
            status != cudaStreamCaptureStatusNone) {
            std::fprintf(stderr,
                         "CUDA cleanup failed: peer stream is still capturing after a discarded "
                         "dual-device capture\n");
        }
    }
    if (restore) { log_cuda_error("cudaSetDevice(restore)", cudaSetDevice(caller_device)); }
}

} // namespace

DecodeGraphPeerBridge::DecodeGraphPeerBridge(int origin_device, int peer_device)
    : origin_device_(origin_device), peer_device_(peer_device) {
    if (origin_device == peer_device) {
        throw std::invalid_argument(
            "DecodeGraphPeerBridge requires two distinct devices; a capture cannot fork a stream "
            "into itself");
    }
    const ScopedCurrentDevice scope;
    cudaEvent_t created[3] = {nullptr, nullptr, nullptr};
    // fork_ is recorded on the origin, join_ and gate_ on the peer.
    const int device_of[3] = {origin_device, peer_device, peer_device};
    for (int slot = 0; slot < 3; ++slot) {
        cudaError_t status = cudaSetDevice(device_of[slot]);
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { (void)cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("DecodeGraphPeerBridge: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    fork_ = created[0];
    join_ = created[1];
    gate_ = created[2];
}

DecodeGraphPeerBridge::~DecodeGraphPeerBridge() {
    destroy_event(fork_);
    destroy_event(join_);
    destroy_event(gate_);
}

DecodeGraphPeerBridge::DecodeGraphPeerBridge(DecodeGraphPeerBridge&& other) noexcept
    : origin_device_(other.origin_device_), peer_device_(other.peer_device_), fork_(other.fork_),
      join_(other.join_), gate_(other.gate_) {
    other.fork_ = nullptr;
    other.join_ = nullptr;
    other.gate_ = nullptr;
}

DecodeGraphPeerBridge& DecodeGraphPeerBridge::operator=(DecodeGraphPeerBridge&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(fork_);
    destroy_event(join_);
    destroy_event(gate_);
    origin_device_ = other.origin_device_;
    peer_device_   = other.peer_device_;
    fork_          = other.fork_;
    join_          = other.join_;
    gate_          = other.gate_;
    other.fork_    = nullptr;
    other.join_    = nullptr;
    other.gate_    = nullptr;
    return *this;
}

void DecodeGraphPeerBridge::gate_launch(cudaStream_t peer_stream,
                                        cudaStream_t origin_stream) const {
    if (gate_ == nullptr) {
        throw std::logic_error("a moved-from DecodeGraphPeerBridge cannot gate a graph launch");
    }
    const ScopedCurrentDevice scope;
    ScopedCurrentDevice::select(peer_device_);
    CUDA_CHECK(cudaEventRecord(gate_, peer_stream));
    ScopedCurrentDevice::select(origin_device_);
    CUDA_CHECK(cudaStreamWaitEvent(origin_stream, gate_, 0));
}

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream, nullptr, false);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body,
                                    const DecodeGraphPeerCapture& peer) {
    if (peer.bridge == nullptr || peer.stream == nullptr) {
        throw std::invalid_argument("dual-device capture requires a peer bridge and stream");
    }

    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    bool peer_forked = false;
    try {
        fork_peer(stream, peer);
        peer_forked = true;
        body();
        join_peer(stream, peer);
    } catch (...) {
        discard_capture(stream, &peer, peer_forked);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

std::size_t DecodeGraphDefinition::node_count() const {
    if (graph_ == nullptr) { return 0; }
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &nodes));
    return nodes;
}

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_       = other.exec_;
    other.exec_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange instantiate_range(nvtx::Name::CudaGraphInstantiate, nvtx::Category::Graph);
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        CUDA_CHECK(err);
    }
    exec_ = exec;
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
            " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    nvtx::ScopedRange upload_range(nvtx::Name::CudaGraphUpload, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphUpload(exec_, stream));
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    // This range executes for every replay; ranges in the captured body execute only at capture.
    nvtx::ScopedRange launch_range(nvtx::Name::CudaGraphLaunch, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace ninfer
