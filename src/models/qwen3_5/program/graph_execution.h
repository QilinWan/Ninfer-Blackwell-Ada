#pragma once
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/execution_context.h"

#include "core/nvtx.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

// A tensor-parallel decode graph holds both ranks' nodes and is launched once, on rank 0's stream.
// Its edges order rank 1's nodes after the graph root, not after work already issued on rank 1's
// own stream (the mirrored KV page and table updates of the round), so the launch is gated on it.
// Captured mailbox all-reduces need no host step per launch: their epoch flags advance with every
// launch on both ranks, and ProgramImpl::synchronize_devices() reports a timed-out exchange.
template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        if (state.execution.tp != nullptr) {
            if (state.execution.graph_bridge == nullptr) {
                throw std::logic_error("tensor-parallel graph launch requires a peer bridge");
            }
            state.execution.graph_bridge->gate_launch(state.execution.tp->execution->dev[1]->stream,
                                                      state.execution.device.stream);
        }
        executable->launch(state.execution.device.stream);
    } else {
        nvtx::ScopedRange eager_range(nvtx::Name::DecodeEager, nvtx::Category::Decode);
        body();
    }
}

// At tensor-parallel width 2 the body issues work on both ranks' streams, ordered by the
// collectives' cross-device events. Those events become graph edges only inside one capture, so
// rank 1's stream is forked into rank 0's capture and joined back before it ends. Both arenas are
// reset first: the body's activations sit at deterministic arena offsets baked into the graph.
template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    if (state.execution.tp == nullptr) {
        definition.capture(state.execution.device.stream, body);
        return;
    }
    if (state.execution.graph_bridge == nullptr) {
        throw std::logic_error("tensor-parallel graph capture requires a peer bridge");
    }
    state.execution.tp->work->reset();
    definition.capture(state.execution.device.stream, body,
                       DecodeGraphPeerCapture{state.execution.graph_bridge,
                                              state.execution.tp->execution->dev[1]->stream});
}

} // namespace ninfer::models::qwen3_5::execution
