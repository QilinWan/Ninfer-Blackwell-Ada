#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <functional>

#include "core/tp2/decode_graph_peer.h"

namespace ninfer {

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    // Dual-device capture: `stream` is the origin (device 0) and `peer` names device 1's stream,
    // which is forked into the same capture for the duration of `body` and joined back before the
    // capture ends. Both `peer.bridge` and `peer.stream` are required.
    void capture(cudaStream_t stream, const std::function<void()>& body,
                 const DecodeGraphPeerCapture& peer);
    [[nodiscard]] bool ready() const noexcept;
    // Node count of the captured graph, 0 when empty. Cross-device event edges are edges, not
    // nodes, so this counts real device work on BOTH devices.
    [[nodiscard]] std::size_t node_count() const;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
