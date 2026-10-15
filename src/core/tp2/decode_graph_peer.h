#pragma once

// The dual-device capture bridge of core/decode_graph.h (two-device tensor parallelism). Our file:
// decode_graph.h includes it with one line; the definitions are in tp2/decode_graph_tp2.inc.

#include <cuda_runtime.h>

namespace ninfer {

// The events that enroll a SECOND device's stream in a capture that began on the first device's
// stream, and that order a replay of the result. Tensor-parallel decode issues work on both
// devices' streams and orders them against each other with cross-device event edges; those edges
// only become graph edges if both streams belong to the SAME capture. Two independent captures
// cannot be linked -- a cudaStreamWaitEvent across two live captures is rejected with
// cudaErrorStreamCaptureMerge -- so a tensor-parallel decode program is ONE graph holding both
// devices' nodes, not one graph per device.
//
//   record(fork) on the origin stream  ->  wait(fork) on the peer stream    (peer joins capture)
//   ...the whole two-device decode body...
//   record(join) on the peer stream    ->  wait(join) on the origin stream  (peer rejoins origin)
//
// The join is mandatory: cudaStreamEndCapture fails with cudaErrorStreamCaptureUnjoined if a
// forked stream is still outstanding. A third and a fourth event serve gate_launch() and
// gate_peer_after_launch() below, which are about replay rather than capture. All four are created
// once (cudaEventCreate is not capturable) and one instance serves an unbounded number of
// sequential captures and launches.
class DecodeGraphPeerBridge {
public:
    DecodeGraphPeerBridge(int origin_device, int peer_device);
    ~DecodeGraphPeerBridge();

    DecodeGraphPeerBridge(const DecodeGraphPeerBridge&)            = delete;
    DecodeGraphPeerBridge& operator=(const DecodeGraphPeerBridge&) = delete;
    DecodeGraphPeerBridge(DecodeGraphPeerBridge&& other) noexcept;
    DecodeGraphPeerBridge& operator=(DecodeGraphPeerBridge&& other) noexcept;

    [[nodiscard]] int origin_device() const noexcept { return origin_device_; }

    [[nodiscard]] int peer_device() const noexcept { return peer_device_; }

    [[nodiscard]] cudaEvent_t fork_event() const noexcept { return fork_; }

    [[nodiscard]] cudaEvent_t join_event() const noexcept { return join_; }

    [[nodiscard]] bool live() const noexcept {
        return fork_ != nullptr && join_ != nullptr && gate_ != nullptr && release_ != nullptr;
    }

    // REPLAY-side ordering, not capture-side. A dual-device graph is launched on the ORIGIN
    // device's stream; the graph's own edges then order its peer-device nodes after the graph
    // root, but they say nothing about work the caller already enqueued on the peer device's own
    // stream (at decode time: the mirrored KV page materialization). Eager execution gets that
    // ordering for free because it issues the peer's kernels on that same stream; a graph launch
    // does not, so the origin stream is explicitly ordered after the peer stream's outstanding
    // work before every launch, which transitively orders the whole graph after it.
    void gate_launch(cudaStream_t peer_stream, cudaStream_t origin_stream) const;

    // The converse, also replay-side. Work issued on the ORIGIN stream after a launch already
    // waits for the whole graph, peer nodes included; work issued on the peer device's own stream
    // does not, since the graph's peer nodes do not run on that stream. Called right after a
    // launch, this orders the peer stream after the whole launched graph, so later peer work (the
    // commit's rank 1 fold, mirrored page and slot updates) cannot overtake the graph's peer
    // nodes even before the caller's host synchronization.
    void gate_peer_after_launch(cudaStream_t peer_stream, cudaStream_t origin_stream) const;

private:
    int origin_device_   = 0;
    int peer_device_     = 0;
    cudaEvent_t fork_    = nullptr;
    cudaEvent_t join_    = nullptr;
    cudaEvent_t gate_    = nullptr;
    cudaEvent_t release_ = nullptr;
};

// The peer half of a dual-device capture: which stream to enroll, and the bridge that enrolls it.
struct DecodeGraphPeerCapture {
    const DecodeGraphPeerBridge* bridge = nullptr;
    cudaStream_t stream                 = nullptr;
};

} // namespace ninfer
