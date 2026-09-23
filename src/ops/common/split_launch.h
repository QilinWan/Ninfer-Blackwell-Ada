#pragma once

// ninfer::ops::detail - shared mechanics for the tp2 split forms of the Op families.
//
// Every split-capable Op family has two extra entry points beside its single-device one:
//
//   <op>_column_parallel  each rank owns a contiguous block of the OUTPUT rows. Rank r reads the
//                         full (replicated) activation and its own weight-row shard and writes its
//                         own output block. No communication; the two blocks concatenate to the
//                         tp1 result.
//   <op>_row_parallel     each rank owns a contiguous block of the INPUT rows. Rank r reads its
//                         own activation block and its own weight-column shard and writes a
//                         FULL-width PARTIAL result; one allreduce_sum then leaves the complete
//                         result on both ranks.
//
// Neither form is a new kernel. A shard is a standalone tensor of the same layout with one axis
// narrowed (that is exactly what the tp2 loader materializes into each device's arena), so the
// existing single-device kernel run against a shard Weight already halves its grid along the
// split axis and already reads only the shard's bytes. What the split forms add is (a) the
// per-rank device/stream discipline below, (b) the cross-rank shape agreement checks a single
// device cannot make, and (c) for the row-parallel form, the collective. The only kernel-side
// change a family needs is that its shape registry admits the shard geometries -- see each
// family's *_config.h / *_dispatch.cpp.
//
// STREAMS. Rank r's work is issued on ec.dev[r]->stream, the same stream include/ninfer/ops/
// allreduce.h runs its collectives on, because "the stream a device executes on" is a property of
// its DeviceContext. The CALLER OBLIGATION documented there applies unchanged and transitively:
// activations staged with the plain cudaMemcpy/cudaMemset/<<<...>>> forms land on the device's
// LEGACY DEFAULT stream, which does NOT implicitly synchronize with DeviceContext::stream, and
// must be retired before a split form reads them.

#include "core/arena.h"  // WorkspaceArena
#include "core/device.h" // DeviceContext, ExecutionContext, CUDA_CHECK
#include "core/device_scope.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::ops::detail {

inline void require_split_context(const ExecutionContext& ec, const char* message) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value() ||
        ec.dev[0]->device == ec.dev[1]->device) {
        throw std::invalid_argument(message);
    }
}

// The weight axis a split form divides across the ranks; they agree on the other one.
enum class SplitAxis : std::uint8_t {
    Output, // column-parallel: rank r holds a block of the output rows N; K agrees.
    Input,  // row-parallel: rank r holds a block of the input columns K; N agrees.
};

// Cross-rank checks every split form makes before either rank issues work, so a rejected pair
// enqueues nothing: two distinct devices and one column extent (`ne[1]`, and `ne[2]` for
// batched activations). Messages start with `op`.
inline void require_split_ranks(const ExecutionContext& ec, const std::array<Tensor, 2>& x,
                                std::string_view op) {
    const std::string prefix(op);
    require_split_context(ec, (prefix + ": requires two distinct devices").c_str());
    if (x[0].ne[1] != x[1].ne[1] || x[0].ne[2] != x[1].ne[2]) {
        throw std::invalid_argument(prefix + ": ranks must agree on the token count");
    }
}

// require_split_ranks() plus one weight format and one extent of the unsplit weight axis.
inline void require_split_pair(const ExecutionContext& ec, const std::array<Tensor, 2>& x,
                               const std::array<Weight, 2>& w, SplitAxis axis,
                               std::string_view op) {
    require_split_ranks(ec, x, op);
    const std::string prefix(op);
    if (w[0].qtype != w[1].qtype || w[0].layout != w[1].layout) {
        throw std::invalid_argument(prefix + ": ranks must agree on the weight format");
    }
    if (axis == SplitAxis::Output && w[0].k != w[1].k) {
        throw std::invalid_argument(prefix + ": ranks must agree on K");
    }
    if (axis == SplitAxis::Input && w[0].n != w[1].n) {
        throw std::invalid_argument(prefix + ": ranks must agree on N");
    }
}

// Rank r's route at this call needs `required_bytes[r]` of transient storage; a rank that needs
// some must have a workspace. Checked with the pair, so rank 0 cannot enqueue work before rank 1
// fails in its dispatch and a row-parallel form skips its all-reduce.
inline void require_split_workspace(const std::array<WorkspaceArena*, 2>& workspace,
                                    const std::array<std::size_t, 2>& required_bytes,
                                    std::string_view op) {
    for (std::size_t rank = 0; rank < 2; ++rank) {
        if (required_bytes[rank] != 0 && workspace[rank] == nullptr) {
            throw std::invalid_argument(std::string(op) +
                                        ": the selected route requires a workspace on every rank");
        }
    }
}

#ifndef NDEBUG
// Debug-only residency predicate, the same one include/ninfer/ops/allreduce.h applies to its own
// buffers and for the same reason: passing rank 1 a pointer that lives on device 0 is the single
// most likely tp2 caller mistake, and it otherwise surfaces as a silently wrong result or an
// opaque launch failure much later. It costs a driver round trip per pointer, so it is compiled
// out of the Release build the product ships.
inline void require_resident_on(const void* pointer, int device, const char* message) {
    if (pointer == nullptr) { return; }
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
        throw std::invalid_argument(message);
    }
}
#endif

// Checks that one rank's activation, weight planes, and output all live on that rank's device.
// A no-op in Release.
inline void require_rank_residency([[maybe_unused]] const ExecutionContext& ec,
                                   [[maybe_unused]] int rank,
                                   [[maybe_unused]] const void* activation,
                                   [[maybe_unused]] const void* weight_payload,
                                   [[maybe_unused]] const void* output,
                                   [[maybe_unused]] const char* message) {
#ifndef NDEBUG
    const int device = ec.dev[rank]->device;
    require_resident_on(activation, device, message);
    require_resident_on(weight_payload, device, message);
    require_resident_on(output, device, message);
#endif
}

// Issues `body(rank)` for rank 0 then rank 1 with that rank's device current, restoring the
// caller's current device afterwards. Rank order is fixed and the calls are enqueue-only, so two
// ranks' kernels overlap on the device even though the host issues them in sequence.
template <class Body>
void for_each_rank(const ExecutionContext& ec, Body&& body) {
    const ScopedCurrentDevice scope;
    for (int rank = 0; rank < 2; ++rank) {
        ScopedCurrentDevice::select(ec.dev[rank]->device);
        body(rank);
    }
}

} // namespace ninfer::ops::detail
