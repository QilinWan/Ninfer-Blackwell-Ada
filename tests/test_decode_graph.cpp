#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_value(void* device, std::uint32_t expected, const char* label) {
    std::uint32_t actual  = 0;
    const cudaError_t err = cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " expected 0x" << std::hex << expected << ", got 0x" << actual << std::dec
              << '\n';
    return 1;
}

int check_external_event(ninfer::DeviceContext& device) {
    ninfer::DeviceArena storage(sizeof(std::uint32_t));
    ninfer::PinnedHostBuffer host(sizeof(std::uint32_t));
    ninfer::CudaCompletionEvent ready(device);
    ninfer::DecodeGraphDefinition definitions[2];
    for (int i = 0; i < 2; ++i) {
        definitions[i].capture(device.stream, [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.base(), i + 1, sizeof(std::uint32_t), device.stream));
            CUDA_CHECK(cudaMemcpyAsync(host.data(), storage.base(), sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, device.stream));
            ready.record_external(device.stream);
            // The handoff exposes the prefix, even though later graph nodes overwrite the source.
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0xff, sizeof(std::uint32_t), device.stream));
        });
    }
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definitions[0]);
    int failures = 0;
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto value = static_cast<std::uint32_t>(iteration % 2 + 1) * 0x01010101U;
        executable.update(definitions[iteration % 2]);
        *static_cast<std::uint32_t*>(host.data()) = 0;
        executable.launch(device.stream);
        ready.synchronize();
        if (*static_cast<std::uint32_t*>(host.data()) != value) {
            std::cerr << "external event exposed stale data after replay/update\n";
            ++failures;
        }
        device.synchronize();
        failures += expect_value(storage.base(), 0xffffffffU, "graph suffix");
    }
    return failures;
}

} // namespace

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << label << '\n';
    return 1;
}

// Registered below with a device-count gate: the peer device is rank 1 of a tensor-parallel
// pair, so a single-device host cannot exercise it; the skip is announced, not silently passed.
bool capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

// One graph holding both devices' nodes: the peer stream is forked into the origin's capture and
// joined back, so a single launch on the origin stream runs both devices' work.
int exercise_dual_device_capture() {
    int failures = 0;
    ninfer::DeviceContext origin(0);
    ninfer::DeviceArena origin_storage(sizeof(std::uint32_t));
    ninfer::DeviceContext peer(1);
    ninfer::DeviceArena peer_storage(sizeof(std::uint32_t));
    origin.bind_to_current_thread();

    const ninfer::DecodeGraphPeerBridge bridge(origin.device, peer.device);
    const ninfer::DecodeGraphPeerCapture peer_capture{.bridge = &bridge, .stream = peer.stream};
    const auto write = [&](int origin_value, int peer_value) {
        CUDA_CHECK(cudaMemsetAsync(origin_storage.base(), origin_value, sizeof(std::uint32_t),
                                   origin.stream));
        peer.bind_to_current_thread();
        CUDA_CHECK(
            cudaMemsetAsync(peer_storage.base(), peer_value, sizeof(std::uint32_t), peer.stream));
        origin.bind_to_current_thread();
    };

    ninfer::DecodeGraphDefinition first;
    first.capture(origin.stream, [&] { write(0x11, 0x44); }, peer_capture);
    ninfer::DecodeGraphDefinition second;
    second.capture(origin.stream, [&] { write(0x22, 0x55); }, peer_capture);
    int current = -1;
    CUDA_CHECK(cudaGetDevice(&current));
    failures += expect(current == origin.device, "dual-device capture changed the current device");
    failures +=
        expect(first.node_count() == 2, "dual-device graph does not hold both devices' nodes");

    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(first);
    executable.launch(origin.stream);
    origin.synchronize();
    failures += expect_value(origin_storage.base(), 0x11111111U, "dual-device origin launch");
    failures += expect_value(peer_storage.base(), 0x44444444U, "dual-device peer launch");

    executable.update(second);
    peer.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(peer_storage.base(), 0x66, sizeof(std::uint32_t), peer.stream));
    origin.bind_to_current_thread();
    bridge.gate_launch(peer.stream, origin.stream);
    executable.launch(origin.stream);
    origin.synchronize();
    failures += expect_value(origin_storage.base(), 0x22222222U, "updated dual-device launch");
    failures += expect_value(peer_storage.base(), 0x55555555U, "gated dual-device peer launch");

    bool body_error = false;
    ninfer::DecodeGraphDefinition discarded;
    try {
        discarded.capture(
            origin.stream,
            [&] {
                write(0x33, 0x77);
                throw std::runtime_error("capture body failure");
            },
            peer_capture);
    } catch (const std::runtime_error&) { body_error = true; }
    failures += expect(body_error && !discarded.ready(), "failed dual-device capture was kept");
    failures += expect(!capturing(origin.stream) && !capturing(peer.stream),
                       "failed dual-device capture left a stream capturing");

    // A memcpy whose source moves from the origin device to the peer keeps the topology but not
    // the node's context: the update is refused with ParametersChanged, the refusal the tp 2
    // decode graphs meet, and update_or_reinstantiate() installs the definition afresh.
    ninfer::DeviceArena origin_source(sizeof(std::uint32_t));
    CUDA_CHECK(cudaMemsetAsync(origin_source.base(), 0x12, sizeof(std::uint32_t), origin.stream));
    peer.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(peer_storage.base(), 0x34, sizeof(std::uint32_t), peer.stream));
    peer.synchronize();
    origin.bind_to_current_thread();
    origin.synchronize();
    ninfer::DecodeGraphDefinition from_origin;
    from_origin.capture(origin.stream, [&] {
        CUDA_CHECK(cudaMemcpyAsync(origin_storage.base(), origin_source.base(),
                                   sizeof(std::uint32_t), cudaMemcpyDefault, origin.stream));
    });
    ninfer::DecodeGraphDefinition from_peer;
    from_peer.capture(origin.stream, [&] {
        CUDA_CHECK(cudaMemcpyAsync(origin_storage.base(), peer_storage.base(),
                                   sizeof(std::uint32_t), cudaMemcpyDefault, origin.stream));
    });
    ninfer::DecodeGraphExecutable copies;
    copies.instantiate(from_origin);
    std::string diagnostic;
    failures += expect(!copies.update_or_reinstantiate(from_peer, diagnostic),
                       "a cross-device parameter change did not fall back to instantiation");
    failures += expect(!diagnostic.empty(), "the instantiation fallback gave no diagnostic");
    copies.launch(origin.stream);
    origin.synchronize();
    failures += expect_value(origin_storage.base(), 0x34343434U, "re-instantiated graph launch");

    bool missing_bridge = false;
    try {
        discarded.capture(origin.stream, [] {}, ninfer::DecodeGraphPeerCapture{});
    } catch (const std::invalid_argument&) { missing_bridge = true; }
    failures += expect(missing_bridge, "dual-device capture accepted a missing peer bridge");
    return failures;
}

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        ninfer::DeviceArena storage(sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x33, sizeof(std::uint32_t), device.stream));
        device.synchronize();

        ninfer::DecodeGraphDefinition first;
        first.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x11, sizeof(std::uint32_t), device.stream));
        });
        ninfer::DecodeGraphDefinition second;
        second.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x22, sizeof(std::uint32_t), device.stream));
        });

        int failures = 0;
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x22222222U, "updated graph launch");

        // A definition the executable cannot take in place (two nodes against one): update()
        // throws naming the result, and so does update_or_reinstantiate(), because a topology
        // change is a profile classification bug, not a driver refusal to patch parameters.
        ninfer::DecodeGraphDefinition reshaped;
        reshaped.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x44, sizeof(std::uint32_t), device.stream));
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x55, sizeof(std::uint32_t), device.stream));
        });
        bool rejected = false;
        try {
            executable.update(reshaped);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("update result") != std::string::npos;
        }
        failures += expect(rejected, "an impossible graph update was not reported");
        std::string diagnostic;
        bool topology_thrown = false;
        try {
            (void)executable.update_or_reinstantiate(reshaped, diagnostic);
        } catch (const std::runtime_error&) { topology_thrown = true; }
        failures += expect(topology_thrown, "a topology change fell back to instantiation");
        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x22222222U,
                                 "a rejected topology change replaced the executable");

        failures += expect(executable.update_or_reinstantiate(second, diagnostic),
                           "a same-topology update fell back to instantiation");

        failures += check_external_event(device);
        // The peer device is tensor-parallel rank 1: a single-device host cannot exercise the
        // dual-device capture, so the gap is announced in the suite output, not passed vacuously.
        if (count >= 2) {
            failures += exercise_dual_device_capture();
        } else {
            std::cout << "SKIP dual-device capture: single device has no peer (TP2 rank 1)\n";
        }

        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
