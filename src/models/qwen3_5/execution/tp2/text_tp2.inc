// Translation-unit fragment of src/models/qwen3_5/execution/text.cpp, included once at its end
// (after its closing namespace): the two-device (tp == 2) schedule of TextContext, every *_tp2
// member and the rank-array overloads declared in text_context_public.inc and
// text_context_private.inc. It shares text.cpp's private helpers (require_tensor_shape,
// ScopedValue, ...) without copying them, and text.cpp's single-device schedule stays upstream's;
// its entry points hand over here with one `if (tp2())` line each. Not a header. Our file; see
// docs/maintainer/upstream-merge.md.

#include "core/device_scope.h"
#include "models/qwen3_5/execution/tp2/split.h"

#include <array>

namespace ninfer::models::qwen3_5::execution {
namespace {

// Issues `body(rank)` for rank 0 then rank 1 with that rank's device current. The calls only
// enqueue, so the two ranks' work overlaps on the devices.
template <class Body>
void for_each_rank(const ExecutionContext& execution, Body&& body) {
    const ScopedCurrentDevice restore;
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        ScopedCurrentDevice::select(execution.dev[static_cast<std::size_t>(rank)]->device);
        body(rank);
    }
}

} // namespace

// =================================================================================================
// Tensor-parallel (tp == 2) forward
// =================================================================================================
//
// Every layer runs the same pattern on both ranks:
//
//   1. replicated elementwise work (RMSNorm, RoPE, gating) per rank, on that rank's stream, over
//      that rank's copy of the replicated residual;
//   2. a column-parallel projection of the rank's heads (attention Q|K|gate|V, GDN A/B and
//      Q|K|V|Z), with no communication;
//   3. head-local mixing: attention of the rank's query heads against its own KV heads and pages,
//      or the GDN convolution and recurrence over its own key/value heads and state slots;
//   4. a row-parallel output projection whose all-reduce folds the residual in once (the mixer's
//      only collective);
//   5. the FFN, whose row-parallel down projection is the layer's second and last collective.
//
// The all-reduce leaves p0 + p1 on rank 0 and p1 + p0 on rank 1, which are the same BF16 value,
// so the residual, and everything derived from it (the next norm, KV pages, GDN state), stays
// identical on both ranks without any other agreement protocol.

void TextContext::validate_tensor_parallel() {
    const TpExecution& tp = *tp_;
    if (!tp.complete() || tp.execution->tp != kTensorParallelWidth || !tp.execution->dev[0] ||
        !tp.execution->dev[1] || !tp.events->live()) {
        throw std::invalid_argument("tensor-parallel TextContext binding is incomplete");
    }
    if (tp.execution->dev[0]->device != ctx_.device ||
        tp.execution->dev[0]->stream != ctx_.stream) {
        throw std::invalid_argument(
            "tensor-parallel TextContext must execute on rank 0 of its ExecutionContext");
    }
    if (&tp.parameters->model != &parameters_.model || parameters_.device != 0 ||
        tp.parameters->device != 1 || parameters_.model.device_count() != kTensorParallelWidth) {
        throw std::invalid_argument(
            "tensor-parallel TextContext requires the rank 0 and rank 1 Parameters of one "
            "two-device Model");
    }
    if (batch_text_kv_ == nullptr) {
        throw std::invalid_argument("tensor-parallel TextContext requires the text KV caches");
    }
    if (tp.linear_attention->slot_count() != state_.slot_count()) {
        throw std::invalid_argument(
            "tensor-parallel Linear Attention state pools disagree on their slots");
    }
    shard_config_       = shard_text_config(config_, kTensorParallelWidth);
    const auto& layers0 = parameters_.text.layers;
    const auto& layers1 = tp.parameters->text.layers;
    if (layers0.size() != layers1.size() || layers0.size() != config_.layer_types.size()) {
        throw std::invalid_argument("tensor-parallel ranks disagree on the Text layers");
    }
    for (std::size_t layer = 0; layer < layers0.size(); ++layer) {
        for (const auto* block : {&layers0[layer], &layers1[layer]}) {
            if (!std::holds_alternative<DenseParameters>(block->ffn)) {
                throw std::invalid_argument("text/layers/" + std::to_string(layer) +
                                            ": the MoE FFN has no tensor-parallel route");
            }
            const ops::ProjectionWeights& projection =
                std::holds_alternative<AttentionParameters>(block->mixer)
                    ? std::get<AttentionParameters>(block->mixer).projection
                    : std::get<GdnParameters>(block->mixer).projection;
            if (!std::holds_alternative<LinearParameters>(projection)) {
                throw std::invalid_argument(
                    "text/layers/" + std::to_string(layer) +
                    ": a paired input projection has no tensor-parallel route; the split "
                    "projections require one contiguous shard parent per rank");
            }
        }
    }
    if (config_.full_attention_layers != 0) {
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const PagedKVBatchLayerView view = rank_text_cache(rank).batch_layer_view(0);
            // The head-local [256,12,2] attention geometry is registered for these two caches
            // only (softmax_attention.h).
            if (view.storage != KvCacheStorage::BFloat16 &&
                view.storage != KvCacheStorage::Int8Group64 &&
                view.storage != KvCacheStorage::Nvfp4Group16) {
                throw std::invalid_argument("tensor-parallel attention supports only BF16, "
                                            "INT8-G64 and NVFP4-G16 KV caches");
            }
            if (view.num_kv_heads != dimension(shard_config_->attention->num_key_value_heads) ||
                view.head_dim != dimension(shard_config_->attention->head_dim)) {
                throw std::invalid_argument(
                    "tensor-parallel KV cache does not hold one rank's KV heads");
            }
        }
    }
    if (mtp_enabled()) { validate_tensor_parallel_mtp(); }
}

void TextContext::validate_tensor_parallel_mtp() {
    const TpExecution& tp = *tp_;
    if (!tp.mtp_complete() || batch_mtp_kv_ == nullptr || !tp.parameters->mtp ||
        mtp_kv_.valid() != tp.mtp_kv.valid()) {
        throw std::invalid_argument("tensor-parallel MTP binding is incomplete");
    }
    if (!io_.mtp || tp.mtp->ar_hidden.ne[0] != io_.mtp->ar_hidden.ne[0] ||
        tp.prefill_hidden.ne[0] != dimension(config_.hidden_size)) {
        throw std::invalid_argument("tensor-parallel MTP frames do not match rank 0's");
    }
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        const MtpParameters& mtp = rank_mtp(rank);
        if (!std::holds_alternative<DenseParameters>(mtp.ffn)) {
            throw std::invalid_argument("tensor-parallel MTP requires the dense post-mixer FFN");
        }
        const PagedKVBatchLayerView view = rank_mtp_cache(rank).batch_layer_view(0);
        if (view.storage != KvCacheStorage::BFloat16 &&
            view.storage != KvCacheStorage::Int8Group64 &&
            view.storage != KvCacheStorage::Nvfp4Group16) {
            throw std::invalid_argument("tensor-parallel MTP attention supports only BF16, "
                                        "INT8-G64 and NVFP4-G16 KV caches");
        }
        if (view.num_kv_heads != dimension(shard_config_->attention->num_key_value_heads) ||
            view.head_dim != dimension(shard_config_->attention->head_dim)) {
            throw std::invalid_argument(
                "tensor-parallel MTP KV cache does not hold one rank's KV heads");
        }
        const auto packed_rows = static_cast<std::uint64_t>(mtp.projection.packed.weight.n);
        if (packed_rows != 2 * shard_config_->attention->query_width() +
                               2 * shard_config_->attention->key_width()) {
            throw std::invalid_argument(
                "tensor-parallel MTP attention projection is not one rank's Q|K|Gate|V shard");
        }
    }
}

void TextContext::require_single_device(const char* operation) const {
    if (tp_ != nullptr) {
        throw std::invalid_argument(std::string(operation) +
                                    " is not implemented at tensor-parallel width 2");
    }
}

const Parameters& TextContext::rank_parameters(int rank) const noexcept {
    return rank == 0 ? parameters_ : *tp_->parameters;
}

std::array<WorkspaceArena*, kTensorParallelWidth> TextContext::workspaces() const noexcept {
    return {&work_, tp_->work};
}

void TextContext::peer_tap_begin(const Tensor& x1) {
    if (peer_feature_sink_ != nullptr) { peer_feature_sink_->begin(x1); }
}

void TextContext::peer_tap_capture_layer(int layer, const Tensor& x1) {
    if (peer_feature_sink_ == nullptr) { return; }
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    peer_feature_sink_->capture_layer(layer, x1, rank_stream(1));
}

void TextContext::peer_tap_capture_positions(const Tensor& positions1) {
    if (peer_feature_sink_ == nullptr) { return; }
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    peer_feature_sink_->capture_positions(positions1, rank_stream(1));
}

void TextContext::peer_tap_consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (peer_feature_sink_ == nullptr) { return; }
    const ScopedCurrentDevice rank1(tp_->execution->dev[1]->device);
    tp_->work->reset();
    peer_feature_sink_->consume_prefill_chunk(tokens, rewrite_checkpoint);
}

cudaStream_t TextContext::rank_stream(int rank) const noexcept {
    return rank == 0 ? ctx_.stream : tp_->execution->dev[1]->stream;
}

DeviceExecutionView TextContext::rank_execution_view(int rank) const noexcept {
    return rank == 0 ? ctx_.execution_view() : tp_->execution->dev[1]->execution_view();
}

LinearAttentionStatePool& TextContext::rank_state(int rank) const noexcept {
    return rank == 0 ? state_ : *tp_->linear_attention;
}

const qwen3_5::PagedKVCache& TextContext::rank_text_cache(int rank) const {
    const qwen3_5::PagedKVCache* cache = rank == 0 ? batch_text_kv_ : tp_->text_cache;
    if (cache == nullptr) { throw std::logic_error("tensor-parallel text KV cache is unbound"); }
    return *cache;
}

const Tensor& TextContext::rank_cache_positions(int rank) const {
    if (rank == 0) {
        return active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    }
    if (peer_cache_positions_ == nullptr) {
        throw std::logic_error("tensor-parallel rank 1 cache positions are unbound");
    }
    return *peer_cache_positions_;
}

const Tensor& TextContext::rank_rope_positions(int rank) const {
    if (rank == 0) {
        return active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    }
    if (peer_rope_positions_ == nullptr) {
        throw std::logic_error("tensor-parallel rank 1 RoPE positions are unbound");
    }
    return *peer_rope_positions_;
}

const Tensor& TextContext::rank_kv_table_rows(int rank) const {
    if (rank == 0) {
        return active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    }
    if (peer_kv_table_rows_ == nullptr) {
        throw std::logic_error("tensor-parallel rank 1 KV table rows are unbound");
    }
    return *peer_kv_table_rows_;
}

const Tensor& TextContext::rank_linear_state_source_slots(int rank) const {
    const Tensor* slots =
        rank == 0 ? active_linear_state_source_slots_ : peer_linear_state_source_slots_;
    if (slots == nullptr) {
        throw std::logic_error("tensor-parallel Linear Attention source slots are unbound");
    }
    return *slots;
}

const Tensor& TextContext::rank_linear_state_destination_slots(int rank) const {
    const Tensor* slots =
        rank == 0 ? active_linear_state_destination_slots_ : peer_linear_state_destination_slots_;
    if (slots == nullptr) {
        throw std::logic_error("tensor-parallel Linear Attention destination slots are unbound");
    }
    return *slots;
}

Tensor TextContext::rank_valid_columns(int rank) const {
    if ((active_valid_columns_ == nullptr) != (peer_valid_columns_ == nullptr)) {
        throw std::logic_error("tensor-parallel valid columns are bound on one rank only");
    }
    const Tensor* valid = rank == 0 ? active_valid_columns_ : peer_valid_columns_;
    return valid != nullptr ? *valid : Tensor{};
}

const Tensor& TextContext::rank_backend_kv_table_rows(int rank) const {
    if ((active_backend_kv_table_rows_ == nullptr) != (peer_backend_kv_table_rows_ == nullptr)) {
        throw std::logic_error("tensor-parallel MTP KV rows are bound on one rank only");
    }
    if (rank == 0) {
        return active_backend_kv_table_rows_ != nullptr ? *active_backend_kv_table_rows_
                                                        : io_.backend_kv_table_row;
    }
    return peer_backend_kv_table_rows_ != nullptr ? *peer_backend_kv_table_rows_
                                                  : tp_->backend_kv_table_row;
}

const qwen3_5::PagedKVCache& TextContext::rank_mtp_cache(int rank) const {
    const qwen3_5::PagedKVCache* cache = rank == 0 ? batch_mtp_kv_ : tp_->mtp_cache;
    if (cache == nullptr) { throw std::logic_error("tensor-parallel MTP KV cache is unbound"); }
    return *cache;
}

qwen3_5::PagedKVCacheView TextContext::rank_mtp_kv(int rank) const {
    const qwen3_5::PagedKVCacheView view = rank == 0 ? mtp_kv_ : tp_->mtp_kv;
    if (!view.valid()) { throw std::logic_error("tensor-parallel MTP prefill KV row is unbound"); }
    return view;
}

const MtpParameters& TextContext::rank_mtp(int rank) const {
    const auto& mtp = rank_parameters(rank).mtp;
    if (!mtp) { throw std::logic_error("tensor-parallel MTP parameters are unavailable"); }
    return *mtp;
}

const GdnReplayRecords& TextContext::rank_replay_records(int rank) const {
    const GdnReplayRecords* records = rank == 0 ? replay_records_ : tp_->replay_records;
    if (records == nullptr) { throw std::logic_error("Replay-record GDN has no record storage"); }
    return *records;
}

void TextContext::attn_mix_tp2(const RankBlocks& w, RankTensors& x, int fidx,
                               const RankTensors& staging) {
    const ExecutionContext& execution = *tp_->execution;
    const AttentionConfig& shard      = *shard_config_->attention;
    const auto ws                     = workspaces();
    const int T                       = x[0].ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }
    const std::array<const AttentionParameters*, 2> p{&std::get<AttentionParameters>(w[0]->mixer),
                                                      &std::get<AttentionParameters>(w[1]->mixer)};
    const std::int32_t head_dim = dimension(shard.head_dim);
    const std::int32_t q_heads  = dimension(shard.num_attention_heads);
    const std::int32_t kv_heads = dimension(shard.num_key_value_heads);
    const std::int32_t q_width  = dimension(shard.query_width());

    RankTensors h;
    RankTensors q_flat;
    RankTensors gate_flat;
    RankTensors k_flat;
    RankTensors v_flat;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto projection = workspace::text_attention_projection(*ws[r], *shard_config_, T);
        h[r]                  = projection.hidden;
        q_flat[r]             = projection.query;
        gate_flat[r]          = projection.gate;
        k_flat[r]             = projection.key;
        v_flat[r]             = projection.value;
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], w[r]->input_norm, config_.rms_norm_eps, true, h[r], rank_stream(rank));
    });
    attention_projection_split(h, p, q_flat, gate_flat, k_flat, v_flat, ws, execution);

    RankTensors qn;
    RankTensors kn;
    RankTensors a;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto results = workspace::text_attention_results(*ws[r], *shard_config_, T);
        qn[r]              = results.normalized_query.view({head_dim, q_heads, T});
        kn[r]              = results.normalized_key.view({head_dim, kv_heads, T});
        a[r]               = results.attention.view({head_dim, q_heads, T});
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    for_each_rank(execution, [&](int rank) {
        const auto r   = static_cast<std::size_t>(rank);
        cudaStream_t s = rank_stream(rank);
        Tensor q       = q_flat[r].view({head_dim, q_heads, T});
        Tensor k       = k_flat[r].view({head_dim, kv_heads, T});
        Tensor v       = v_flat[r].view({head_dim, kv_heads, T});
        Tensor gate    = gate_flat[r].view({head_dim, q_heads, T});
        ops::rmsnorm(q, p[r]->query_norm, config_.rms_norm_eps, true, qn[r], s);
        ops::rmsnorm(k, p[r]->key_norm, config_.rms_norm_eps, true, kn[r], s);
        const Tensor& cache_positions = rank_cache_positions(rank);
        const Tensor& rope_positions  = rank_rope_positions(rank);
        Tensor rope_for_op =
            active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
        text_rope(rope_for_op, *config_.rope_parameters, qn[r], kn[r], s);

        const PagedKVBatchLayerView cache = rank_text_cache(rank).batch_layer_view(fidx);
        const Tensor& kv_table_rows       = rank_kv_table_rows(rank);
        if (active_sequence_batch_ != 0) {
            const std::int32_t width = active_sequence_width_;
            const std::int32_t batch = active_sequence_batch_;
            if (width <= 0 || width * batch != T) {
                throw std::logic_error(
                    "Text sequence batch binding does not match aggregate columns");
            }
            Tensor q_batch        = qn[r].view({head_dim, q_heads, width, batch});
            Tensor k_batch        = kn[r].view({head_dim, kv_heads, width, batch});
            Tensor v_batch        = v.view({head_dim, kv_heads, width, batch});
            Tensor a_batch        = a[r].view({head_dim, q_heads, width, batch});
            Tensor position_batch = cache_positions.view({width, batch});
            ops::causal_softmax_attention(q_batch, k_batch, v_batch, position_batch,
                                          rank_valid_columns(rank), kv_table_rows,
                                          {head_dim, q_heads, kv_heads}, scale, cache,
                                          *active_causal_attention_envelope_, *ws[r], a_batch, s);
        } else {
            ops::causal_softmax_attention(qn[r], kn[r], v, cache_positions, Tensor{}, kv_table_rows,
                                          {head_dim, q_heads, kv_heads}, scale, cache,
                                          *active_causal_attention_envelope_, *ws[r], a[r], s);
        }
        ops::sigmoid_mul(gate, a[r], s);
    });

    attention_output_split({a[0].view({q_width, T}), a[1].view({q_width, T})}, p, x, staging, ws,
                           execution, *tp_->events);
}

void TextContext::gdn_mix_tp2(const RankBlocks& w, RankTensors& x, int gidx, Phase ph,
                              const RankTensors& staging) {
    const ExecutionContext& execution = *tp_->execution;
    const GdnConfig& shard            = *shard_config_->gdn;
    const auto ws                     = workspaces();
    const int T                       = x[0].ne[1];
    const std::array<const GdnParameters*, 2> p{&std::get<GdnParameters>(w[0]->mixer),
                                                &std::get<GdnParameters>(w[1]->mixer)};
    const std::int32_t key_dim     = dimension(shard.linear_key_head_dim);
    const std::int32_t key_heads   = dimension(shard.linear_num_key_heads);
    const std::int32_t value_dim   = dimension(shard.linear_value_head_dim);
    const std::int32_t value_heads = dimension(shard.linear_num_value_heads);
    const std::int32_t key_width   = dimension(shard.key_width());
    const std::int32_t value_width = dimension(shard.value_width());
    const auto layer               = static_cast<std::uint32_t>(gidx);

    RankTensors h;
    RankTensors g;
    RankTensors beta;
    RankTensors z;
    RankTensors qc;
    RankTensors kc;
    RankTensors vc;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto control    = workspace::gdn_control(*ws[r], *shard_config_, T);
        h[r]                  = control.hidden;
        g[r]                  = control.g;
        beta[r]               = control.beta;
        const auto projection = workspace::gdn_projection(*ws[r], *shard_config_, T);
        z[r]                  = projection.output_gate;
        qc[r]                 = projection.query;
        kc[r]                 = projection.key;
        vc[r]                 = projection.value;
    }
    // The single-device path fuses this norm into the gating GEMM, which has no split form. The
    // norm is replicated elementwise work, so it runs per rank ahead of the split gating.
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], w[r]->input_norm, config_.rms_norm_eps, true, h[r], rank_stream(rank));
    });
    gdn_control_split(h, p, g, beta, ws, execution);

    if (ph == Phase::Verify) {
        const std::int32_t width = active_sequence_width_;
        const std::int32_t batch = active_sequence_batch_;
        if (batch == 0 || width <= 0 || width * batch != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        const bool record = gdn_state_action_ == GdnStateAction::RecordForReplay;
        if (!record && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        RankTensors projection_input;
        RankTensors query_output;
        RankTensors key_output;
        RankTensors value_output;
        RankTensors gate_output;
        RankTensors conv_states;
        RankTensors valid;
        RankTensors source_slots;
        for (std::size_t r = 0; r < 2; ++r) {
            const int rank      = static_cast<int>(r);
            projection_input[r] = h[r].view({dimension(config_.hidden_size), width, batch});
            query_output[r]     = qc[r].view({key_width, width, batch});
            key_output[r]       = kc[r].view({key_width, width, batch});
            value_output[r]     = vc[r].view({value_width, width, batch});
            gate_output[r]      = z[r].view({value_width, width, batch});
            conv_states[r]      = rank_state(rank).layer_view(layer).conv;
            valid[r]            = rank_valid_columns(rank);
            source_slots[r]     = rank_linear_state_source_slots(rank);
        }
        if (record) {
            // Speculative verification: each rank records its own heads' convolution inputs and
            // recurrence, and the Program folds the accepted prefix into each rank's state.
            RankTensors conv_record;
            for (std::size_t r = 0; r < 2; ++r) {
                conv_record[r] = rank_replay_records(static_cast<int>(r)).layer(gidx, batch).conv;
            }
            gdn_projection_record_split(projection_input, p, conv_states, valid, source_slots,
                                        conv_record, query_output, key_output, value_output,
                                        gate_output, ws, execution);
        } else {
            RankTensors destination_slots;
            for (std::size_t r = 0; r < 2; ++r) {
                destination_slots[r] = rank_linear_state_destination_slots(static_cast<int>(r));
            }
            gdn_projection_snapshot_split(projection_input, p, conv_states, valid, source_slots,
                                          destination_slots, query_output, key_output, value_output,
                                          gate_output, ws, execution);
        }
    } else {
        RankTensors qkv;
        for (std::size_t r = 0; r < 2; ++r) {
            qkv[r] = workspace::gdn_prefill_conv(*ws[r], *shard_config_, T);
        }
        gdn_projection_split(h, p, qkv, z, ws, execution);
        for_each_rank(execution, [&](int rank) {
            const auto r         = static_cast<std::size_t>(rank);
            Tensor conv_state_in = rank_state(rank).conv_slot(layer, linear_state_source_slot_);
            Tensor conv_state_out =
                rank_state(rank).conv_slot(layer, linear_state_destination_slot_);
            ops::causal_conv1d_silu_split(qkv[r], p[r]->convolution, conv_state_in, conv_state_out,
                                          qc[r], kc[r], vc[r], rank_stream(rank));
        });
    }

    RankTensors o;
    for (std::size_t r = 0; r < 2; ++r) {
        o[r] = workspace::gdn_recurrent_output(*ws[r], *shard_config_, T)
                   .view({value_dim, value_heads, T});
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(key_dim)));
    for_each_rank(execution, [&](int rank) {
        const auto r       = static_cast<std::size_t>(rank);
        cudaStream_t s     = rank_stream(rank);
        Tensor q_recurrent = qc[r].view({key_dim, key_heads, T});
        Tensor k_recurrent = kc[r].view({key_dim, key_heads, T});
        Tensor v_recurrent = vc[r].view({value_dim, value_heads, T});
        if (ph == Phase::Verify) {
            const std::int32_t width = active_sequence_width_;
            const std::int32_t batch = active_sequence_batch_;
            Tensor recurrent_states  = rank_state(rank).layer_view(layer).recurrent;
            Tensor q_batch           = q_recurrent.view({key_dim, key_heads, width, batch});
            Tensor k_batch           = k_recurrent.view({key_dim, key_heads, width, batch});
            Tensor v_batch           = v_recurrent.view({value_dim, value_heads, width, batch});
            Tensor g_batch           = g[r].view({value_heads, width, batch});
            Tensor beta_batch        = beta[r].view({value_heads, width, batch});
            Tensor out_batch         = o[r].view({value_dim, value_heads, width, batch});
            if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
                GdnReplayRecordLayer records = rank_replay_records(rank).layer(gidx, batch);
                ops::gated_delta_net_replay_record(
                    q_batch, k_batch, v_batch, g_batch, beta_batch, scale, recurrent_states,
                    rank_valid_columns(rank), rank_linear_state_source_slots(rank), records.key,
                    records.value, records.gate, out_batch, s);
            } else {
                ops::gated_delta_net_batch_update(
                    q_batch, k_batch, v_batch, g_batch, beta_batch, scale,
                    /*normalize_qk=*/true, recurrent_states, rank_linear_state_source_slots(rank),
                    rank_linear_state_destination_slots(rank), out_batch, s);
            }
        } else {
            Tensor recurrent_state_in =
                rank_state(rank).recurrent_slot(layer, linear_state_source_slot_);
            Tensor recurrent_state_out =
                rank_state(rank).recurrent_slot(layer, linear_state_destination_slot_);
            ops::gated_delta_net(q_recurrent, k_recurrent, v_recurrent, g[r], beta[r], scale,
                                 /*normalize_qk=*/true, *ws[r], recurrent_state_in,
                                 recurrent_state_out, o[r], rank_execution_view(rank));
        }
    });

    RankTensors on;
    for (std::size_t r = 0; r < 2; ++r) {
        on[r] = workspace::gdn_normalized_output(*ws[r], *shard_config_, T)
                    .view({value_dim, value_heads, T});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        Tensor gate  = z[r].view({value_dim, value_heads, T});
        // The norm weight is per head dimension and replicated, so each rank applies all of it
        // over its own value heads.
        ops::gated_rmsnorm(o[r], p[r]->norm, gate, config_.rms_norm_eps, on[r], rank_stream(rank));
    });

    gdn_output_split({on[0].view({value_width, T}), on[1].view({value_width, T})}, p, x, staging,
                     ws, execution, *tp_->events);
}

void TextContext::mlp_tail_tp2(const RankBlocks& w, RankTensors& x, const RankTensors& staging) {
    const ExecutionContext& execution = *tp_->execution;
    const auto ws                     = workspaces();
    RankTensors h;
    for (std::size_t r = 0; r < 2; ++r) {
        h[r] = workspace::post_mixer_hidden(*ws[r], config_, x[r].ne[1]);
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], w[r]->post_attention_norm, config_.rms_norm_eps, true, h[r],
                     rank_stream(rank));
    });
    ffn_split(h, {&w[0]->ffn, &w[1]->ffn}, x, staging, ws, execution, *tp_->events);
}

template <class Tap>
void TextContext::run_layers_tp2(RankTensors& x, Phase ph, const RankTensors& staging, Tap& tap) {
    const bool prefill  = ph == Phase::Prefill;
    const auto& layers0 = parameters_.text.layers;
    const auto& layers1 = tp_->parameters->text.layers;
    for (std::size_t layer = 0; layer < layers0.size(); ++layer) {
        const RankBlocks block{&layers0[layer], &layers1[layer]};
        const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
        const auto compact = dimension(config_.compact_layer_indices[layer]);
        nvtx::ScopedRange layer_range(
            full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
                 : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        try {
            {
                nvtx::ScopedRange mixer_range(
                    full ? (prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention)
                         : (prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn),
                    full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
                auto scope0 = work_.scope();
                auto scope1 = tp_->work->scope();
                if (full) {
                    attn_mix_tp2(block, x, compact, staging);
                } else {
                    gdn_mix_tp2(block, x, compact, ph, staging);
                }
            }
            {
                nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer
                                                : nvtx::Name::VerifyPostMixer,
                                        nvtx::Category::PostMixer, layer);
                auto scope0 = work_.scope();
                auto scope1 = tp_->work->scope();
                mlp_tail_tp2(block, x, staging);
            }
            if constexpr (Tap::enabled) {
                tap.capture_layer(static_cast<int>(layer), x[0], ctx_.stream);
                peer_tap_capture_layer(static_cast<int>(layer), x[1]);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "text/layers/" + std::to_string(layer) + (prefill ? " prefill" : " verify") +
                " tp2 columns=" + std::to_string(x[0].ne[1]) + ": " + error.what());
        }
    }
}

void TextContext::logits_tp2(const RankTensors& hidden, Tensor& logits) {
    logits_tp2(hidden, {&parameters_.text.output_head, &tp_->parameters->text.output_head}, logits);
}

void TextContext::logits_tp2(const RankTensors& hidden,
                             const std::array<const LinearParameters*, 2>& head, Tensor& logits) {
    const auto ws              = workspaces();
    const std::int32_t columns = hidden[0].ne[1];
    const std::int32_t rows0   = head[0]->weight.n;
    const std::int32_t rows1   = head[1]->weight.n;
    auto scope0                = work_.scope();
    auto scope1                = tp_->work->scope();
    const auto part0           = workspace::tp_logits(*ws[0], rows0, rows1, columns, true);
    const auto part1           = workspace::tp_logits(*ws[1], rows1, rows0, columns, false);
    output_logits_split_rank0(hidden, head, {part0.partial, part1.partial}, logits, part0.staging,
                              ws, *tp_->execution, *tp_->events);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl_tp2(std::span<const int> ids, const TextPrefill* text_prefill,
                              const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    if ((text_prefill == nullptr) == (multimodal == nullptr)) {
        throw std::logic_error(
            "tensor-parallel prefill takes exactly one text or multimodal prompt");
    }
    const std::uint32_t base = text_kv_base_;
    if (text_prefill != nullptr &&
        (base != text_prefill->begin ||
         text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size())) {
        throw std::invalid_argument("text prefill chunk does not match its full prompt");
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    // The prompt the MTP alignment reads its shifted ids from.
    const std::span<const int> alignment_ids =
        multimodal != nullptr ? multimodal->token_ids : text_prefill->token_ids;
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    cudaStream_t s  = ctx_.stream;
    const int T     = static_cast<int>(ids.size());
    const int chunk = static_cast<int>(prefill_chunk_);
    // Each rank offsets its own positions by its own copy of the delta.
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);
    {
        const ScopedCurrentDevice peer(execution.dev[1]->device);
        Tensor peer_rope_delta = tp_->rope_delta;
        ops::set_i32_scalar(peer_rope_delta, rope_delta_, rank_stream(1));
    }
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;

    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }

    // One chunk per call, exactly as the single-device path: the caller advances the frontier.
    int len = std::min(chunk, T);
    if (split_rel > 0 && len > split_rel) { len = split_rel; }
    // The Vision session caps the chunk at the next item and, on an item's first chunk, encodes it
    // on the tower's rank and copies the merged embeddings to the other rank.
    VisionChunk vision_chunk;
    if (multimodal != nullptr) {
        vision_chunk = multimodal->vision->prepare_chunk(base, static_cast<std::uint32_t>(len));
        len          = vision_chunk.length;
        if (len <= 0 || len > T) {
            throw std::logic_error("Vision chunk length is outside the prefill chunk");
        }
    }
    const bool is_last = finalize_at_end && len == T;
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    const auto ws = workspaces();
    work_.reset();
    tp_->work->reset();
    {
        // The chunk's visual columns, relative to the chunk, and their first merged token.
        std::vector<std::int32_t> local_scatter_indices;
        std::int32_t visual_begin = 0;
        if (vision_chunk.control != nullptr) {
            if (vision_chunk.peer_embeddings.data == nullptr) {
                throw std::logic_error("tensor-parallel Vision chunk has no rank 1 embeddings");
            }
            const auto scatter =
                std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
            const auto begin = std::lower_bound(scatter.begin(), scatter.end(), base_i);
            const auto end   = std::lower_bound(begin, scatter.end(), base_i + len);
            visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
            local_scatter_indices.reserve(static_cast<std::size_t>(end - begin));
            for (auto index = begin; index != end; ++index) {
                local_scatter_indices.push_back(*index - base_i);
            }
        }
        const auto visual_columns = static_cast<std::int32_t>(local_scatter_indices.size());
        // A multimodal chunk reads its three-axis positions from the prompt. A text chunk
        // continuing a multimodal prefix keeps that prefix's RoPE delta, as on one device; each
        // rank offsets its own positions with its own copy of the delta.
        const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
        std::vector<std::int32_t> rope_positions_host;
        if (multimodal != nullptr) {
            rope_positions_host.resize(static_cast<std::size_t>(3) * len);
            const std::size_t prompt_tokens = multimodal->token_ids.size();
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = multimodal->positions.data() +
                                  static_cast<std::size_t>(axis) * prompt_tokens + base;
                std::copy_n(src, len,
                            rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
            }
        }
        std::array<workspace::TextPrefillRoots, 2> roots;
        RankTensors staging;
        for (std::size_t r = 0; r < 2; ++r) {
            roots[r] =
                workspace::text_prefill_roots(*ws[r], config_, len, rope_axes, visual_columns);
            staging[r] = workspace::tp_call_roots(*ws[r], config_, len).staging;
        }
        RankTensors positions;
        RankTensors rope_positions;
        RankTensors x;
        for_each_rank(execution, [&](int rank) {
            const auto r        = static_cast<std::size_t>(rank);
            cudaStream_t stream = rank_stream(rank);
            copy_i32(ids.data(), roots[r].ids, stream);
            positions[r] = roots[r].positions;
            ops::fill_i32_positions(positions[r], base_i, stream);
            rope_positions[r] = positions[r];
            if (multimodal != nullptr) {
                rope_positions[r] = roots[r].rope_positions;
                copy_i32(rope_positions_host.data(), rope_positions[r], stream);
            } else if (rope_delta_ != 0) {
                rope_positions[r] = roots[r].rope_positions;
                ops::offset_i32_positions(positions[r],
                                          rank == 0 ? io_.rope_delta : tp_->rope_delta,
                                          rope_positions[r], stream);
            }
            x[r] = roots[r].residual;
            ops::embedding(roots[r].ids, rank_parameters(rank).text.token_embedding, x[r], stream);
            if (visual_columns != 0) {
                // The hidden axis is replicated: both ranks scatter the same merged embeddings,
                // each from its own copy.
                Tensor indices = roots[r].scatter_indices;
                copy_i32(local_scatter_indices.data(), indices, stream);
                const Tensor& merged =
                    rank == 0 ? vision_chunk.embeddings : vision_chunk.peer_embeddings;
                Tensor embeddings = merged.slice(1, visual_begin, visual_columns);
                ops::scatter(embeddings, indices, x[r], stream);
            }
        });

        ScopedPositions scoped_cache(active_cache_positions_, positions[0]);
        ScopedPositions scoped_rope(active_rope_positions_, rope_positions[0]);
        ScopedValue<const Tensor*> peer_cache(peer_cache_positions_, &positions[1]);
        ScopedValue<const Tensor*> peer_rope(peer_rope_positions_, &rope_positions[1]);
        ScopedValue<const Tensor*> peer_rows(peer_kv_table_rows_, &tp_->text_kv_table_row);
        const auto visible = static_cast<std::uint32_t>(base_i + len);
        const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
        ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

        if constexpr (Tap::enabled) {
            tap.begin(x[0]);
            peer_tap_begin(x[1]);
        }
        run_layers_tp2(x, Phase::Prefill, staging, tap);
        if constexpr (requires { tap.capture_positions(positions[0], s); }) {
            tap.capture_positions(positions[0], s);
            peer_tap_capture_positions(positions[1]);
        }

        // Rank 0 keeps the whole normalized chunk (prefill_hidden and the rewrite checkpoint read
        // it). Rank 1 keeps it too when the MTP head consumes the chunk, whose input projection
        // half on rank 1 contracts the normalized hidden; otherwise it only needs the last column,
        // as its input to the vocabulary-split head.
        Tensor xf = prefill_hidden_.data != nullptr
                        ? matrix_window(prefill_hidden_, len)
                        : work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
        ops::rmsnorm(x[0], *final_norm_, config_.rms_norm_eps, true, xf, s);
        Tensor peer_xf;
        if (prepare_mtp_prompt) {
            Tensor peer_hidden = tp_->prefill_hidden;
            peer_xf            = matrix_window(peer_hidden, len);
            const ScopedCurrentDevice peer(execution.dev[1]->device);
            ops::rmsnorm(x[1], rank_parameters(1).text.final_norm, config_.rms_norm_eps, true,
                         peer_xf, rank_stream(1));
        }

        if (is_last) {
            Tensor peer_last;
            if (prepare_mtp_prompt) {
                peer_last = peer_xf.slice(1, len - 1, 1);
            } else {
                peer_last = ws[1]->alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                const ScopedCurrentDevice peer(execution.dev[1]->device);
                ops::rmsnorm(x[1].slice(1, len - 1, 1), rank_parameters(1).text.final_norm,
                             config_.rms_norm_eps, true, peer_last, rank_stream(1));
            }
            Tensor logits = matrix_window(io_.logits, 1);
            logits_tp2({xf.slice(1, len - 1, 1), peer_last}, logits);
            // Sampling belongs to rank 0 alone, over the gathered complete logits.
            ops::set_i32_scalar(io_.pos, base_i + T, s);
            ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
            if (sampling_config_ != nullptr) {
                ops::sample(logits, io_.token,
                            dimension(parameters_.model.resources().public_token_count),
                            sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
            } else {
                ops::argmax(logits, io_.token,
                            dimension(parameters_.model.resources().public_token_count), s);
            }
        }

        if (prepare_mtp_prompt) {
            // The MTP head consumes the shifted token stream against the text model's own final
            // hidden, one column behind the target, exactly as on one device. Only rank 0 needs
            // the shifted ids: its input projection half contracts the token embedding.
            const auto alignment_tokens = static_cast<std::uint32_t>(alignment_ids.size());
            const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                alignment_tokens, base, static_cast<std::uint32_t>(len));
            const int prompt_columns =
                len - static_cast<int>(mtp_window.final_column_uses_generated_token);
            Tensor mtp_ids = work_.alloc(DType::I32, {len});
            if (prompt_columns != 0) {
                Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin, prompt_mtp_ids,
                         s);
            }
            if (mtp_window.final_column_uses_generated_token) {
                Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
            // A multimodal prompt's shifted stream carries visual columns: rank 0 embeds the
            // shifted ids and scatters the item's merged embeddings over the shifted visual
            // overlap, as on one device. Rank 1's stem half contracts the hidden alone.
            Tensor mtp_input_embeddings;
            const Tensor* mtp_input_embeddings_ptr = nullptr;
            if (multimodal != nullptr) {
                mtp_input_embeddings =
                    work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
                ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                if (vision_chunk.control != nullptr) {
                    const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                        vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                    if (!overlap.empty()) {
                        Tensor shifted_indices = workspace::visual_scatter_indices(
                            work_, static_cast<std::int32_t>(overlap.size()));
                        qwen3_5::detail::scatter_shifted_visual_embeddings(
                            mtp_input_embeddings, vision_chunk.embeddings, overlap, shifted_indices,
                            s);
                    }
                }
                mtp_input_embeddings_ptr = &mtp_input_embeddings;
            }
            const RankTensors mtp_input{xf, peer_xf};
            if (is_last && mtp_proposal_extent_ != 0) {
                Tensor logits = matrix_window(io_.logits, 1);
                Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                const RankTensors ar_hidden{io_.mtp->ar_hidden, tp_->mtp->ar_hidden};
                mtp_prefill_chunk_tp2(mtp_ids, mtp_input, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &ar_hidden, &logits,
                                      &draft0);

                const RankTensors ar_position{io_.mtp->position.slice(0, 0, 1),
                                              tp_->mtp->position.slice(0, 0, 1)};
                for_each_rank(execution, [&](int rank) {
                    Tensor position = ar_position[static_cast<std::size_t>(rank)];
                    ops::set_i32_scalar(position, base_i + T, rank_stream(rank));
                });
                for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                    auto step_scope0      = work_.scope();
                    auto step_scope1      = tp_->work->scope();
                    Tensor previous_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                    Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                    RankTensors next_hidden;
                    for (std::size_t r = 0; r < 2; ++r) {
                        next_hidden[r] =
                            ws[r]->alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                    }
                    const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                    const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                    mtp_forward_ar_step(previous_token, ar_hidden, ar_position, ar_envelope,
                                        next_hidden, logits, next_token);
                    for_each_rank(execution, [&](int rank) {
                        const auto r        = static_cast<std::size_t>(rank);
                        cudaStream_t stream = rank_stream(rank);
                        CUDA_CHECK(cudaMemcpyAsync(ar_hidden[r].data, next_hidden[r].data,
                                                   ar_hidden[r].bytes(), cudaMemcpyDeviceToDevice,
                                                   stream));
                        Tensor position = ar_position[r];
                        ops::increment_i32_scalar(position, stream);
                    });
                }
            } else {
                mtp_prefill_chunk_tp2(mtp_ids, mtp_input, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
            }
        }

        if (split_rel > 0 && len == split_rel && rewrite_checkpoint_hidden_output_ != nullptr) {
            require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                 {dimension(config_.hidden_size), 1},
                                 "rewrite checkpoint hidden output");
            const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
            CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                       checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, s));
            if (peer_rewrite_checkpoint_hidden_output_ != nullptr) {
                // Rank 1 retains the same column of its own final-normed chunk: the hidden axis is
                // replicated, so this is a local copy on rank 1, not a transfer.
                if (!prepare_mtp_prompt) {
                    throw std::logic_error(
                        "rank 1 retains a checkpoint hidden only with its MTP prompt chunk");
                }
                require_tensor_shape(*peer_rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {dimension(config_.hidden_size), 1},
                                     "rank 1 rewrite checkpoint hidden output");
                const Tensor peer_checkpoint_hidden = peer_xf.slice(1, len - 1, 1);
                const ScopedCurrentDevice peer(execution.dev[1]->device);
                CUDA_CHECK(cudaMemcpyAsync(
                    peer_rewrite_checkpoint_hidden_output_->data, peer_checkpoint_hidden.data,
                    peer_checkpoint_hidden.bytes(), cudaMemcpyDeviceToDevice, rank_stream(1)));
            }
        }
    }

    if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
        work_.reset();
        tap.consume_prefill_chunk(len, split_rel > 0 && len == split_rel);
        peer_tap_consume_prefill_chunk(len, split_rel > 0 && len == split_rel);
    }

    prefill_split_frontier_ = -1;

    timing.begin_wait();
    ctx_.synchronize();
    execution.dev[1]->synchronize();
    timing.end_wait();
    work_.reset();
    tp_->work->reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(len),
                              .finalized        = finalize_at_end && len == T,
                              .timing           = timing.finish()};
}

// =================================================================================================
// Tensor-parallel (tp == 2) speculative verification and MTP
// =================================================================================================
//
// Verification is the text schedule above over K+1 columns per row, with masked columns and
// ReplaySSM records on each rank. The MTP head is one decoder layer, so its split schedule is a
// text layer's with the MTP's operands:
//
//   rank 0: e = rmsnorm(embed(ids), embedding_norm)   rank 1: h = rmsnorm(hidden, hidden_norm)
//   x         = all_reduce(fc_0 e + fc_1 h)          row-parallel over the packed [e; h] input
//   q|k|g|v_r = split(packed_r rmsnorm(x))           column-parallel, rank r's heads
//   a_r       = gated attention over rank r's MTP KV heads and pages
//   x        += all_reduce(o_r a_r)                  row-parallel, then the residual add
//   x        += all_reduce(down_r swiglu(gate|up_r)) column- then row-parallel
//   out       = rmsnorm(x, final_norm)               replicated
//
// Rank 0's packed-input half is the normalized embedding and rank 1's the normalized hidden, so
// neither rank forms the packed [2 * hidden, T] input and rank 1 never embeds a token. Each
// all-reduce leaves the identical BF16 residual on both ranks, which keeps both ranks' MTP KV pages
// and the next call's inputs in agreement.

template <class Tap>
void TextContext::target_verify_batch_tp2_impl(
    const RankTensors& ids, const RankTensors& cache_positions, const RankTensors& rope_positions,
    const RankTensors& valid_columns, const RankTensors& kv_table_rows,
    const RankTensors& linear_state_source_slots, ops::CausalAttentionExecutionEnvelope envelope,
    const RankTensors& hidden, Tensor& logits, Tensor& target_tokens, Tap& tap) {
    if (!tp2()) {
        throw std::invalid_argument("rank-pair target verification requires tensor parallelism");
    }
    const std::int32_t width = ids[0].ne[0];
    const std::int32_t batch = ids[0].ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    const std::int32_t H       = dimension(config_.hidden_size);
    const std::int32_t V       = dimension(config_.vocab_size);
    // Rank 1's destinations live on the other device, where an undersized buffer would be a silent
    // out-of-bounds write, so both ranks' extents are checked. The logits are rank 0's alone.
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(ids[r], DType::I32, {width, batch}, "target verify batch ids");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "target verify batch cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "target verify batch RoPE positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "target verify batch valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch}, "target verify batch KV rows");
        require_tensor_shape(linear_state_source_slots[r], DType::I32, {batch},
                             "target verify batch Linear Attention slots");
        require_tensor_shape(hidden[r], DType::BF16, {H, width, batch},
                             "target verify batch hidden");
    }
    require_tensor_shape(logits, DType::BF16, {V, width, batch}, "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws = workspaces();
    work_.reset();
    tp_->work->reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions[0]);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions[0]);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows[0]);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots[0]);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
        ScopedValue<const Tensor*> peer_cache_binding(peer_cache_positions_, &cache_positions[1]);
        ScopedValue<const Tensor*> peer_rope_binding(peer_rope_positions_, &rope_positions[1]);
        ScopedValue<const Tensor*> peer_rows_binding(peer_kv_table_rows_, &kv_table_rows[1]);
        ScopedValue<const Tensor*> peer_state_binding(peer_linear_state_source_slots_,
                                                      &linear_state_source_slots[1]);
        ScopedValue<const Tensor*> peer_valid_binding(peer_valid_columns_, &valid_columns[1]);

        RankTensors x;
        RankTensors staging;
        for (std::size_t r = 0; r < 2; ++r) {
            x[r]       = ws[r]->alloc(DType::BF16, {H, columns});
            staging[r] = workspace::tp_call_roots(*ws[r], config_, columns).staging;
        }
        for_each_rank(execution, [&](int rank) {
            const auto r          = static_cast<std::size_t>(rank);
            const Tensor flat_ids = ids[r].view({columns});
            ops::embedding(flat_ids, rank_parameters(rank).text.token_embedding, x[r],
                           rank_stream(rank));
        });
        // A DFlash feature tap reads rank 0's residual after each layer's all-reduce, which holds
        // the complete hidden state, and scatters it into rank 0's pending features; the split
        // drafter's peer sink does the same with rank 1's copy.
        if constexpr (Tap::enabled) {
            tap.begin(x[0]);
            peer_tap_begin(x[1]);
        }
        run_layers_tp2(x, Phase::Verify, staging, tap);
        if constexpr (requires { tap.capture_positions(cache_positions[0], ctx_.stream); }) {
            tap.capture_positions(cache_positions[0], ctx_.stream);
            peer_tap_capture_positions(cache_positions[1]);
        }

        RankTensors flat_hidden;
        for (std::size_t r = 0; r < 2; ++r) { flat_hidden[r] = hidden[r].view({H, columns}); }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], rank_parameters(rank).text.final_norm, config_.rms_norm_eps, true,
                         flat_hidden[r], rank_stream(rank));
        });
        Tensor flat_logits = logits.view({V, columns});
        logits_tp2(flat_hidden, {&parameters_.text.output_head, &tp_->parameters->text.output_head},
                   flat_logits);
        // Acceptance is rank 0's alone, so only rank 0 needs the target tokens.
        Tensor flat_tokens = target_tokens.view({columns});
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
    work_.reset();
    tp_->work->reset();
}

void TextContext::target_verify_batch(
    const RankTensors& ids, const RankTensors& cache_positions, const RankTensors& rope_positions,
    const RankTensors& valid_columns, const RankTensors& kv_table_rows,
    const RankTensors& linear_state_source_slots, ops::CausalAttentionExecutionEnvelope envelope,
    const RankTensors& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_tp2_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                                 linear_state_source_slots, envelope, hidden, logits, target_tokens,
                                 tap);
}

void TextContext::target_verify_batch(
    const RankTensors& ids, const RankTensors& cache_positions, const RankTensors& rope_positions,
    const RankTensors& valid_columns, const RankTensors& kv_table_rows,
    const RankTensors& linear_state_source_slots, ops::CausalAttentionExecutionEnvelope envelope,
    const RankTensors& hidden, Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink) {
    target_verify_batch_tp2_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                                 linear_state_source_slots, envelope, hidden, logits, target_tokens,
                                 sink);
}

void TextContext::mtp_forward_stem_tp2(const Tensor& ids, const RankTensors& hidden,
                                       const Tensor* input_embeddings, RankTensors& x,
                                       RankTensors& ah, const RankTensors& staging) {
    const ExecutionContext& execution = *tp_->execution;
    const auto ws                     = workspaces();
    const std::int32_t H              = dimension(config_.hidden_size);
    const std::int32_t T              = ids.ne[0] * ids.ne[1];
    // The aggregate column count is what must match: the prompt and proposal calls pass [H,T],
    // the decode batch [H,W,B].
    for (std::size_t r = 0; r < 2; ++r) {
        if (hidden[r].dtype != DType::BF16 || !hidden[r].is_contiguous() ||
            hidden[r].data == nullptr || hidden[r].ne[0] != H ||
            hidden[r].numel() != static_cast<std::int64_t>(H) * T) {
            throw std::invalid_argument(
                "tensor-parallel MTP stem hidden does not match the token count");
        }
    }
    // Only rank 0 embeds tokens, or takes their composed embeddings (a multimodal prompt's, with
    // its visual columns scattered in); the plan sizes both ranks for rank 0's roots.
    if (input_embeddings != nullptr &&
        (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != H ||
         input_embeddings->numel() != static_cast<std::int64_t>(H) * T ||
         !input_embeddings->is_contiguous() || input_embeddings->data == nullptr)) {
        throw std::invalid_argument("tensor-parallel MTP input embeddings shape mismatch");
    }
    std::array<workspace::MtpStemSplitRoots, 2> roots{
        workspace::mtp_stem_split(*ws[0], config_, T, input_embeddings == nullptr),
        workspace::mtp_stem_split(*ws[1], config_, T, false)};
    for (std::size_t r = 0; r < 2; ++r) {
        x[r]  = roots[r].residual;
        ah[r] = roots[r].attention_hidden;
    }
    const Tensor flat_ids = ids.view({T});
    for_each_rank(execution, [&](int rank) {
        cudaStream_t stream      = rank_stream(rank);
        const MtpParameters& mtp = rank_mtp(rank);
        if (rank == 0) {
            Tensor embedding;
            if (input_embeddings != nullptr) {
                embedding = input_embeddings->view({H, T});
            } else {
                embedding = roots[0].embedding;
                ops::embedding(flat_ids, *embed_, embedding, stream);
            }
            ops::rmsnorm(embedding, mtp.embedding_norm, config_.rms_norm_eps, true,
                         roots[0].normalized_input, stream);
        } else {
            ops::rmsnorm(hidden[1].view({H, T}), mtp.hidden_norm, config_.rms_norm_eps, true,
                         roots[1].normalized_input, stream);
        }
    });
    mtp_input_projection_split({roots[0].normalized_input, roots[1].normalized_input},
                               {&rank_mtp(0), &rank_mtp(1)}, x, staging, ws, execution,
                               *tp_->events);
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], rank_mtp(rank).input_norm, config_.rms_norm_eps, true, ah[r],
                     rank_stream(rank));
    });
}

void TextContext::mtp_forward_tail_tp2(RankTensors& x, const RankTensors& ah,
                                       const RankTensors& positions,
                                       const RankTensors& rope_positions,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       const RankTensors& mtp_hidden, const RankTensors& staging) {
    const ExecutionContext& execution = *tp_->execution;
    const AttentionConfig& shard      = *shard_config_->attention;
    const auto ws                     = workspaces();
    const std::int32_t T              = x[0].ne[1];
    const std::int32_t H              = dimension(config_.hidden_size);
    const std::int32_t head_dim       = dimension(shard.head_dim);
    const std::int32_t q_heads        = dimension(shard.num_attention_heads);
    const std::int32_t kv_heads       = dimension(shard.num_key_value_heads);
    const std::int32_t q_width        = dimension(shard.query_width());

    RankTensors q_flat;
    RankTensors gate_flat;
    RankTensors k_flat;
    RankTensors v_flat;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto projection = workspace::mtp_attention_projection(*ws[r], *shard_config_, T);
        q_flat[r]             = projection.query;
        gate_flat[r]          = projection.gate;
        k_flat[r]             = projection.key;
        v_flat[r]             = projection.value;
    }
    mtp_projection_split(ah, {&rank_mtp(0).projection, &rank_mtp(1).projection}, shard, q_flat,
                         gate_flat, k_flat, v_flat, ws, execution);

    RankTensors qn;
    RankTensors kn;
    RankTensors a;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto results = workspace::mtp_attention_results(*ws[r], *shard_config_, T);
        qn[r]              = results.normalized_query.view({head_dim, q_heads, T});
        kn[r]              = results.normalized_key.view({head_dim, kv_heads, T});
        a[r]               = results.attention.view({head_dim, q_heads, T});
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    for_each_rank(execution, [&](int rank) {
        const auto r             = static_cast<std::size_t>(rank);
        cudaStream_t stream      = rank_stream(rank);
        const MtpParameters& mtp = rank_mtp(rank);
        Tensor q                 = q_flat[r].view({head_dim, q_heads, T});
        Tensor k                 = k_flat[r].view({head_dim, kv_heads, T});
        Tensor v                 = v_flat[r].view({head_dim, kv_heads, T});
        Tensor gate              = gate_flat[r].view({head_dim, q_heads, T});
        ops::rmsnorm(q, mtp.query_norm, config_.rms_norm_eps, true, qn[r], stream);
        ops::rmsnorm(k, mtp.key_norm, config_.rms_norm_eps, true, kn[r], stream);
        Tensor rope_for_op =
            active_sequence_batch_ != 0 ? rope_positions[r].view({T}) : rope_positions[r];
        text_rope(rope_for_op, *config_.rope_parameters, qn[r], kn[r], stream);

        const PagedKVBatchLayerView cache = rank_mtp_cache(rank).batch_layer_view(0);
        const Tensor& rows                = rank_backend_kv_table_rows(rank);
        if (active_sequence_batch_ != 0) {
            const std::int32_t width = active_sequence_width_;
            const std::int32_t batch = active_sequence_batch_;
            if (width <= 0 || width * batch != T) {
                throw std::logic_error("MTP sequence batch binding is incomplete");
            }
            Tensor q_batch        = qn[r].view({head_dim, q_heads, width, batch});
            Tensor k_batch        = kn[r].view({head_dim, kv_heads, width, batch});
            Tensor v_batch        = v.view({head_dim, kv_heads, width, batch});
            Tensor a_batch        = a[r].view({head_dim, q_heads, width, batch});
            Tensor position_batch = positions[r].view({width, batch});
            ops::causal_softmax_attention(
                q_batch, k_batch, v_batch, position_batch, rank_valid_columns(rank), rows,
                {head_dim, q_heads, kv_heads}, scale, cache, envelope, *ws[r], a_batch, stream);
        } else {
            ops::causal_softmax_attention(qn[r], kn[r], v, positions[r], Tensor{}, rows,
                                          {head_dim, q_heads, kv_heads}, scale, cache, envelope,
                                          *ws[r], a[r], stream);
        }
        ops::sigmoid_mul(gate, a[r], stream);
    });

    // The MTP output projection is a plain linear plus the residual add, as on one device.
    RankTensors o;
    RankTensors mh;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto post = workspace::mtp_post_attention(*ws[r], config_, T);
        o[r]            = post.output;
        mh[r]           = post.post_mixer_hidden;
    }
    mtp_output_split({a[0].view({q_width, T}), a[1].view({q_width, T})},
                     {&rank_mtp(0), &rank_mtp(1)}, o, staging, ws, execution, *tp_->events);
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t stream = rank_stream(rank);
        ops::residual_add(o[r], x[r], stream);
        ops::rmsnorm(x[r], rank_mtp(rank).post_attention_norm, config_.rms_norm_eps, true, mh[r],
                     stream);
    });
    {
        auto post_mixer_scope0 = work_.scope();
        auto post_mixer_scope1 = tp_->work->scope();
        mtp_ffn_split(mh, {&rank_mtp(0).ffn, &rank_mtp(1).ffn}, x, staging, ws, execution,
                      *tp_->events);
    }
    for_each_rank(execution, [&](int rank) {
        const auto r           = static_cast<std::size_t>(rank);
        Tensor flat_mtp_hidden = mtp_hidden[r].view({H, T});
        ops::rmsnorm(x[r], rank_mtp(rank).final_norm, config_.rms_norm_eps, true, flat_mtp_hidden,
                     rank_stream(rank));
    });
}

void TextContext::mtp_forward_core_tp2(const Tensor& ids, const RankTensors& hidden,
                                       const RankTensors& positions,
                                       const RankTensors& rope_positions,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       const RankTensors& mtp_hidden,
                                       const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr || tp_->mtp_cache == nullptr) {
        throw std::runtime_error("MTP forward is not enabled");
    }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    const auto ws        = workspaces();
    auto scratch_scope0  = work_.scope();
    auto scratch_scope1  = tp_->work->scope();
    const std::int32_t T = ids.ne[0] * ids.ne[1];
    // One staging per rank serves the three all-reduces, each of which leaves it free on return.
    RankTensors staging;
    for (std::size_t r = 0; r < 2; ++r) {
        staging[r] = workspace::tp_call_roots(*ws[r], config_, T).staging;
    }
    RankTensors x;
    RankTensors ah;
    mtp_forward_stem_tp2(ids, hidden, input_embeddings, x, ah, staging);
    mtp_forward_tail_tp2(x, ah, positions, rope_positions, envelope, mtp_hidden, staging);
}

void TextContext::mtp_prefill_chunk_tp2(const Tensor& ids, const RankTensors& hidden,
                                        const Tensor* input_embeddings,
                                        const RankTensors& positions,
                                        const RankTensors& rope_positions,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        bool final_chunk, const RankTensors* final_hidden,
                                        Tensor* logits, Tensor* draft_token) {
    if (!mtp_kv_.valid() || !tp_->mtp_kv.valid()) {
        throw std::runtime_error("MTP prefill is not enabled");
    }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    const std::int32_t H = dimension(config_.hidden_size);
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {H, T}, "MTP prefill hidden");
        require_tensor_shape(positions[r], DType::I32, {T}, "MTP prefill positions");
        if (rope_positions[r].dtype != DType::I32 || rope_positions[r].ne[0] != T ||
            rope_positions[r].ne[1] != rope_positions[0].ne[1] ||
            (rope_positions[r].ne[1] != 1 && rope_positions[r].ne[1] != 3) ||
            rope_positions[r].ne[2] != 1 || rope_positions[r].ne[3] != 1 ||
            !rope_positions[r].is_contiguous() || rope_positions[r].data == nullptr) {
            throw std::invalid_argument(
                "tensor-parallel MTP prefill rope positions must be [T] or [T,3] on both ranks");
        }
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        for (std::size_t r = 0; r < 2; ++r) {
            require_tensor_shape((*final_hidden)[r], DType::BF16, {H, 1},
                                 "MTP final prefill hidden");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    const ExecutionContext& execution = *tp_->execution;
    const AttentionConfig& shard      = *shard_config_->attention;
    const auto ws                     = workspaces();
    const std::int32_t head_dim       = dimension(shard.head_dim);
    const std::int32_t q_heads        = dimension(shard.num_attention_heads);
    const std::int32_t kv_heads       = dimension(shard.num_key_value_heads);
    const std::array<const MtpProjectionParameters*, 2> projection{&rank_mtp(0).projection,
                                                                   &rank_mtp(1).projection};
    auto scratch_scope0 = work_.scope();
    auto scratch_scope1 = tp_->work->scope();
    RankTensors staging;
    RankTensors last_staging;
    RankTensors x_last;
    RankTensors ah_last;
    for (std::size_t r = 0; r < 2; ++r) {
        staging[r] = workspace::tp_call_roots(*ws[r], config_, T).staging;
        if (final_chunk) {
            // The final column's all-reduces reduce one column; their staging matches it.
            last_staging[r] = ws[r]->alloc(DType::BF16, {H, 1});
            x_last[r]       = ws[r]->alloc(DType::BF16, {H, 1});
            ah_last[r]      = ws[r]->alloc(DType::BF16, {H, 1});
        }
    }

    {
        auto bulk_scope0 = work_.scope();
        auto bulk_scope1 = tp_->work->scope();
        RankTensors x;
        RankTensors ah;
        mtp_forward_stem_tp2(ids, hidden, input_embeddings, x, ah, staging);

        RankTensors k_flat;
        RankTensors v_flat;
        RankTensors kn;
        for (std::size_t r = 0; r < 2; ++r) {
            k_flat[r] = ws[r]->alloc(DType::BF16, {dimension(shard.key_width()), T});
            v_flat[r] = ws[r]->alloc(DType::BF16, {dimension(shard.key_width()), T});
        }
        mtp_kv_projection_split(ah, projection, shard, k_flat, v_flat, ws, execution);
        for (std::size_t r = 0; r < 2; ++r) {
            kn[r] = ws[r]->alloc(DType::BF16, {head_dim, kv_heads, T});
        }
        const std::size_t column_bytes = static_cast<std::size_t>(H) * dtype_size(DType::BF16);
        for_each_rank(execution, [&](int rank) {
            const auto r        = static_cast<std::size_t>(rank);
            cudaStream_t stream = rank_stream(rank);
            Tensor k            = k_flat[r].view({head_dim, kv_heads, T});
            Tensor v            = v_flat[r].view({head_dim, kv_heads, T});
            ops::rmsnorm(k, rank_mtp(rank).key_norm, config_.rms_norm_eps, true, kn[r], stream);
            text_rope(rope_positions[r], *config_.rope_parameters, kn[r], stream);
            ops::kv_cache_append(kn[r], v, positions[r], rank_mtp_kv(rank).layer_view(0), stream);
            if (final_chunk) {
                const auto* x_src  = static_cast<const unsigned char*>(x[r].data) +
                                     static_cast<std::size_t>(T - 1) * column_bytes;
                const auto* ah_src = static_cast<const unsigned char*>(ah[r].data) +
                                     static_cast<std::size_t>(T - 1) * column_bytes;
                CUDA_CHECK(cudaMemcpyAsync(x_last[r].data, x_src, column_bytes,
                                           cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(ah_last[r].data, ah_src, column_bytes,
                                           cudaMemcpyDeviceToDevice, stream));
            }
        });
    }

    if (!final_chunk) { return; }

    RankTensors q_flat;
    RankTensors gate_flat;
    RankTensors qn;
    RankTensors a;
    RankTensors o;
    RankTensors mh;
    for (std::size_t r = 0; r < 2; ++r) {
        q_flat[r]    = ws[r]->alloc(DType::BF16, {dimension(shard.query_width()), 1});
        gate_flat[r] = ws[r]->alloc(DType::BF16, {dimension(shard.query_width()), 1});
    }
    mtp_query_gate_projection_split(ah_last, projection, shard, q_flat, gate_flat, ws, execution);
    // A multimodal chunk's three-axis positions are axis-major, so the last column's three values
    // are gathered into [1,3], as on one device.
    const bool three_axis_rope = rope_positions[0].ne[1] == 3;
    RankTensors last_rope;
    for (std::size_t r = 0; r < 2; ++r) {
        qn[r]        = ws[r]->alloc(DType::BF16, {head_dim, q_heads, 1});
        a[r]         = ws[r]->alloc(DType::BF16, {head_dim, q_heads, 1});
        o[r]         = ws[r]->alloc(DType::BF16, {H, 1});
        mh[r]        = ws[r]->alloc(DType::BF16, {H, 1});
        last_rope[r] = three_axis_rope ? ws[r]->alloc(DType::I32, {1, 3})
                                       : rope_positions[r].slice(0, T - 1, 1);
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t stream = rank_stream(rank);
        Tensor q            = q_flat[r].view({head_dim, q_heads, 1});
        Tensor gate         = gate_flat[r].view({head_dim, q_heads, 1});
        ops::rmsnorm(q, rank_mtp(rank).query_norm, config_.rms_norm_eps, true, qn[r], stream);
        const Tensor last_position = positions[r].slice(0, T - 1, 1);
        if (three_axis_rope) {
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions[r].data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst       = static_cast<std::int32_t*>(last_rope[r].data) + axis;
                CUDA_CHECK(cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice,
                                           stream));
            }
        }
        text_rope(last_rope[r], *config_.rope_parameters, qn[r], stream);
        ops::causal_softmax_attention_cached(qn[r], last_position, {head_dim, q_heads, kv_heads},
                                             scale, rank_mtp_kv(rank).layer_view(0), envelope,
                                             *ws[r], a[r], stream);
        ops::sigmoid_mul(gate, a[r], stream);
    });
    const std::int32_t q_width = dimension(shard.query_width());
    mtp_output_split({a[0].view({q_width, 1}), a[1].view({q_width, 1})},
                     {&rank_mtp(0), &rank_mtp(1)}, o, last_staging, ws, execution, *tp_->events);
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t stream = rank_stream(rank);
        ops::residual_add(o[r], x_last[r], stream);
        ops::rmsnorm(x_last[r], rank_mtp(rank).post_attention_norm, config_.rms_norm_eps, true,
                     mh[r], stream);
    });
    {
        auto post_mixer_scope0 = work_.scope();
        auto post_mixer_scope1 = tp_->work->scope();
        mtp_ffn_split(mh, {&rank_mtp(0).ffn, &rank_mtp(1).ffn}, x_last, last_staging, ws, execution,
                      *tp_->events);
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        Tensor out   = (*final_hidden)[r];
        ops::rmsnorm(x_last[r], rank_mtp(rank).final_norm, config_.rms_norm_eps, true, out,
                     rank_stream(rank));
    });
    proposal_argmax_tp2(*final_hidden, *logits, *draft_token);
}

void TextContext::proposal_argmax_tp2(const RankTensors& hidden, Tensor& logits,
                                      Tensor& proposal_tokens) {
    const std::int32_t T = hidden[0].ne[1];
    const std::int32_t H = dimension(config_.hidden_size);
    const std::int32_t V = dimension(config_.vocab_size);
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {H, T}, "proposal hidden");
    }
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, V, T, "proposal logits");
    if (proposal_head_ != nullptr) {
        const std::optional<ProposalParameters>& peer = tp_->parameters->proposal;
        if (!peer) {
            // A PrimaryOnly optimized head: rank 0 projects, picks and maps alone.
            proposal_argmax(hidden[0], logits, proposal_tokens);
            return;
        }
        // Split by vocabulary: each rank projects and picks within its rows, rank 0 combines the
        // two candidates and maps the selected row to its token.
        nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                         static_cast<std::uint64_t>(T));
        const std::int32_t rows = proposal_head_->weight.n;
        if (proposal_head_ids_ == nullptr || peer->head.weight.n != rows ||
            2 * rows != proposal_head_n_) {
            throw std::logic_error("tensor-parallel proposal head split is inconsistent");
        }
        const auto ws = workspaces();
        auto scope0   = work_.scope();
        auto scope1   = tp_->work->scope();
        const std::array<workspace::TensorParallelProposalRoots, 2> roots{
            workspace::tp_proposal_argmax(*ws[0], rows, T),
            workspace::tp_proposal_argmax(*ws[1], rows, T)};
        proposal_argmax_split(hidden, {proposal_head_, &peer->head},
                              {roots[0].partial, roots[1].partial},
                              {roots[0].local, roots[1].local},
                              {roots[0].candidates, roots[1].candidates},
                              {roots[0].staging, roots[1].staging}, proposal_tokens, ws,
                              *tp_->execution, *tp_->events);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
        return;
    }
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    Tensor output_logits = matrix_window(logits, T);
    // The full head is the vocabulary-split output head: the argmax runs over the gathered
    // complete logits, on rank 0.
    logits_tp2(hidden, {&rank_mtp(0).output_head, &rank_mtp(1).output_head}, output_logits);
    ops::argmax(output_logits, proposal_tokens,
                dimension(parameters_.model.resources().public_token_count), ctx_.stream);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const RankTensors& hidden,
                                           const RankTensors& cache_positions,
                                           const RankTensors& rope_positions,
                                           const RankTensors& valid_columns,
                                           const RankTensors& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           const RankTensors& mtp_hidden) {
    if (!tp2()) { throw std::invalid_argument("rank-pair MTP decode requires tensor parallelism"); }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    const std::int32_t H = dimension(config_.hidden_size);
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {H, width, batch},
                             "MTP decode batch target hidden");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "MTP decode batch cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "MTP decode batch RoPE positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "MTP decode batch valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch}, "MTP decode batch KV rows");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {H, width, batch},
                             "MTP decode batch hidden");
    }
    const ScopedCurrentDevice device(ctx_.device);
    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows[0]);
    ScopedValue<const Tensor*> peer_backend_binding(peer_backend_kv_table_rows_, &kv_table_rows[1]);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
    ScopedValue<const Tensor*> peer_valid_binding(peer_valid_columns_, &valid_columns[1]);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core_tp2(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden,
                         nullptr);
}

void TextContext::mtp_propose_batch(const RankTensors& hidden, Tensor& logits,
                                    Tensor& draft_tokens) {
    if (!tp2()) {
        throw std::invalid_argument("rank-pair MTP proposal requires tensor parallelism");
    }
    const std::int32_t batch = hidden[0].ne[1];
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {dimension(config_.hidden_size), batch},
                             "MTP proposal batch hidden");
    }
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    const ScopedCurrentDevice device(ctx_.device);
    auto proposal_scope0 = work_.scope();
    auto proposal_scope1 = tp_->work->scope();
    proposal_argmax_tp2(hidden, logits, draft_tokens);
}

void TextContext::mtp_forward_batch(const Tensor& ids, const RankTensors& hidden,
                                    const RankTensors& positions, const RankTensors& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    const RankTensors& mtp_hidden, int logits_column,
                                    Tensor* logits, Tensor* draft_token,
                                    const Tensor* input_embeddings) {
    if (!tp2()) { throw std::invalid_argument("rank-pair MTP bridge requires tensor parallelism"); }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    const std::int32_t H = dimension(config_.hidden_size);
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(positions[r], DType::I32, {T}, "MTP positions");
        if (rope_positions[r].ne[1] == 3) {
            require_tensor_shape(rope_positions[r], DType::I32, {T, 3}, "MTP RoPE positions");
        } else {
            require_tensor_shape(rope_positions[r], DType::I32, {T}, "MTP RoPE positions");
        }
        require_tensor_shape(hidden[r], DType::BF16, {H, T}, "MTP hidden");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {H, T}, "MTP output hidden");
    }
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }
    const ScopedCurrentDevice device(ctx_.device);
    // Unbatched: each rank appends and attends through its own prefill MTP row (io_ on rank 0,
    // TpExecution::backend_kv_table_row on rank 1), as the single-device bridge does.
    mtp_forward_core_tp2(ids, hidden, positions, rope_positions, envelope, mtp_hidden,
                         input_embeddings);
    if (logits_column >= 0) {
        auto logits_scope0 = work_.scope();
        auto logits_scope1 = tp_->work->scope();
        const RankTensors column{mtp_hidden[0].slice(1, logits_column, 1),
                                 mtp_hidden[1].slice(1, logits_column, 1)};
        proposal_argmax_tp2(column, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const RankTensors& previous_hidden,
                                      const RankTensors& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      const RankTensors& mtp_hidden, Tensor& logits,
                                      Tensor& draft_token) {
    if (!tp2()) {
        throw std::invalid_argument("rank-pair MTP proposal step requires tensor parallelism");
    }
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t H = dimension(config_.hidden_size);
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(position[r], DType::I32, {1}, "MTP AR position");
        require_tensor_shape(previous_hidden[r], DType::BF16, {H, 1}, "MTP AR previous hidden");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {H, 1}, "MTP AR output hidden");
    }
    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws        = workspaces();
    auto position_scope0 = work_.scope();
    auto position_scope1 = tp_->work->scope();
    RankTensors rope_position;
    for (std::size_t r = 0; r < 2; ++r) { rope_position[r] = ws[r]->alloc(DType::I32, {1}); }
    // Each rank offsets its own copy of the position by its own copy of the sequence's RoPE delta,
    // which the Program and the prompt chunks keep equal on both ranks.
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::offset_i32_positions(position[r], rank == 0 ? io_.rope_delta : tp_->rope_delta,
                                  rope_position[r], rank_stream(rank));
    });
    mtp_forward_core_tp2(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                         nullptr);
    auto logits_scope0 = work_.scope();
    auto logits_scope1 = tp_->work->scope();
    proposal_argmax_tp2(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch_tp2(const Tensor& ids, const Tensor& cache_positions,
                                            const Tensor& rope_positions,
                                            const Tensor& kv_table_rows,
                                            const Tensor& linear_state_source_slots,
                                            const Tensor& linear_state_destination_slots,
                                            ops::CausalAttentionExecutionEnvelope envelope,
                                            Tensor& hidden, Tensor& logits) {
    if (tp_->ordinary == nullptr) {
        throw std::logic_error("tensor-parallel decode requires rank 1's ordinary frame");
    }
    const std::int32_t batch = ids.ne[0];
    const auto peer_window   = [batch](const Tensor& source, const char* label) {
        if (source.dtype != DType::I32 || source.ne[0] < batch || source.ne[1] != 1 ||
            source.ne[2] != 1 || source.ne[3] != 1 || !source.is_contiguous() ||
            source.data == nullptr) {
            throw std::invalid_argument(std::string("tensor-parallel rank 1 ordinary ") + label +
                                        " does not cover the batch");
        }
        return source.slice(0, 0, batch);
    };
    const OrdinaryPeerFrame& frame  = *tp_->ordinary;
    const Tensor peer_ids           = peer_window(frame.tokens, "tokens");
    const Tensor peer_cache         = peer_window(frame.cache_positions, "cache positions");
    const Tensor peer_rope          = peer_window(frame.rope_positions, "RoPE positions");
    const Tensor peer_rows          = peer_window(frame.text_kv_table_rows, "KV rows");
    const Tensor peer_state_sources = peer_window(frame.state_source_slots, "source slots");
    const Tensor peer_state_destinations =
        peer_window(frame.state_destination_slots, "destination slots");

    const ExecutionContext& execution = *tp_->execution;
    const ScopedCurrentDevice device(ctx_.device);
    const auto ws = workspaces();
    work_.reset();
    tp_->work->reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);
        ScopedValue<const Tensor*> peer_cache_binding(peer_cache_positions_, &peer_cache);
        ScopedValue<const Tensor*> peer_rope_binding(peer_rope_positions_, &peer_rope);
        ScopedValue<const Tensor*> peer_rows_binding(peer_kv_table_rows_, &peer_rows);
        ScopedValue<const Tensor*> peer_source_binding(peer_linear_state_source_slots_,
                                                       &peer_state_sources);
        ScopedValue<const Tensor*> peer_destination_binding(peer_linear_state_destination_slots_,
                                                            &peer_state_destinations);

        RankTensors x;
        RankTensors staging;
        for (std::size_t r = 0; r < 2; ++r) {
            x[r]       = ws[r]->alloc(DType::BF16, {dimension(config_.hidden_size), batch});
            staging[r] = workspace::tp_call_roots(*ws[r], config_, batch).staging;
        }
        const std::array<const Tensor*, 2> rank_ids{&ids, &peer_ids};
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::embedding(*rank_ids[r], rank_parameters(rank).text.token_embedding, x[r],
                           rank_stream(rank));
        });
        NullTap tap;
        run_layers_tp2(x, Phase::Verify, staging, tap);

        // Rank 0's normalized hidden is the call's output; rank 1's is only the input of its
        // vocabulary half.
        ops::rmsnorm(x[0], *final_norm_, config_.rms_norm_eps, true, hidden, ctx_.stream);
        Tensor peer_hidden = ws[1]->alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        {
            const ScopedCurrentDevice peer(execution.dev[1]->device);
            ops::rmsnorm(x[1], rank_parameters(1).text.final_norm, config_.rms_norm_eps, true,
                         peer_hidden, rank_stream(1));
        }
        logits_tp2({hidden, peer_hidden}, logits);
    }
    work_.reset();
    tp_->work->reset();
}

} // namespace ninfer::models::qwen3_5::execution
