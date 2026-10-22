// Translation-unit fragment of src/models/qwen3_5/execution/draft.cpp, included once at its end
// (after its closing namespace): rank 1's peer operands of the split drafter, so the code below
// shares that file's private append (append_context_impl). Not a header.

#include "models/qwen3_5/execution/tp2/draft_split.h"

namespace ninfer::models::qwen3_5::execution {

void dflash_append_context_peer(DFlashPeerDrafter& peer, const qwen3_5::DFlashDecodeIngress& host,
                                std::int32_t batch, std::uint32_t draft_window,
                                ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    ExecutionCore& core = peer.execution;
    if (!core.io.dflash_decode) {
        throw std::logic_error("DFlash2 tp2: rank 1 has no DFlash decode frame");
    }
    const ScopedCurrentDevice rank1(core.device.device);
    qwen3_5::DFlashDecodeState& frame = *core.io.dflash_decode;
    const cudaStream_t stream         = core.device.stream;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &host, sizeof(qwen3_5::DFlashDecodeIngress),
                               cudaMemcpyHostToDevice, stream));
    core.work.reset();
    Tensor features =
        core.work.alloc(DType::BF16, {dimension(core.parameters.draft->feature_projection.weight.k),
                                      static_cast<std::int32_t>(draft_window + 1U), batch});
    Tensor positions = frame.append_positions.slice(1, 0, batch);
    Tensor counts    = frame.append_counts.slice(0, 0, batch);
    ops::prepare_ragged_prefix(peer.dflash.pending_features, frame.active_lanes.slice(0, 0, batch),
                               frame.context_frontiers.slice(0, 0, batch),
                               frame.execution_frontiers.slice(0, 0, batch), features, positions,
                               counts, stream);
    DFlashAppendContext append{core, peer.dflash};
    append_context_impl(append, features, positions, counts,
                        frame.state_destination_slots.slice(0, 0, batch),
                        frame.dflash_kv_table_rows.slice(0, 0, batch), envelope);
}

DFlashFeatureSink dflash_peer_batch_sink(DFlashPeerDrafter& peer, const Tensor& lanes,
                                         const Tensor& valid_columns, std::int32_t width,
                                         std::int32_t batch) {
    const auto& config = *peer.execution.parameters.model.config().draft;
    return DFlashFeatureSink{
        .batch_features      = &peer.dflash.pending_features,
        .batch_lanes         = &lanes,
        .batch_valid_columns = &valid_columns,
        .batch_width         = width,
        .batch_size          = batch,
        .layers              = std::span<const std::uint32_t>(config.target_layer_ids),
    };
}

DFlashFeatureSink dflash_peer_prefill_sink(DFlashPeerDrafter& peer,
                                           qwen3_5::DFlashPrefillIngress* host_ingress) {
    const auto& config = *peer.execution.parameters.model.config().draft;
    return DFlashFeatureSink{
        .features  = &peer.dflash.prefill_features,
        .positions = &peer.dflash.prefill_positions,
        .layers    = std::span<const std::uint32_t>(config.target_layer_ids),
        .consume_prefill =
            [&peer, host_ingress](const Tensor& features, const Tensor& positions, bool) {
                ExecutionCore& core = peer.execution;
                if (!core.io.dflash_prefill || host_ingress == nullptr) {
                    throw std::logic_error("DFlash2 tp2: rank 1 prefill controls are unavailable");
                }
                auto& frame = *core.io.dflash_prefill;
                // Rank 0's consumer filled the host record with this chunk's bindings.
                CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, host_ingress,
                                           sizeof(qwen3_5::DFlashPrefillIngress),
                                           cudaMemcpyHostToDevice, core.device.stream));
                const auto exact = static_cast<std::uint32_t>(features.ne[1]);
                DFlashAppendContext append{core, peer.dflash};
                append_context_impl(append, features, positions, frame.append_count,
                                    frame.state_destination_slot, frame.full_kv_table_row,
                                    {exact, exact});
            },
    };
}

} // namespace ninfer::models::qwen3_5::execution
