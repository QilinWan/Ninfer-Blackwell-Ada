#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "models/qwen3_5/execution/visual_scatter.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

// Current-device save/restore around per-rank issue: a kernel launch or a stream-ordered memcpy
// targets the current device, and the caller's device is restored afterwards.
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(device));
    }

    DeviceScope(const DeviceScope&)            = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

    ~DeviceScope() { (void)cudaSetDevice(previous_); }

private:
    int previous_ = 0;
};

// Issues `body(rank)` for rank 0 then rank 1 with that rank's device current. The calls only
// enqueue, so the two ranks' work overlaps on the devices.
template <class Body>
void for_each_rank(const ExecutionContext& execution, Body&& body) {
    const DeviceScope restore(execution.dev[0]->device);
    for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[static_cast<std::size_t>(rank)]->device));
        body(rank);
    }
}

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const execution::Parameters& weights,
                         WorkspaceArena& work, qwen3_5::PagedKVCacheView kv,
                         LinearAttentionStatePool& state, qwen3_5::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, qwen3_5::PagedKVCacheView mtp_kv,
                         const qwen3_5::PagedKVCache* batch_text_kv,
                         const qwen3_5::PagedKVCache* batch_mtp_kv, const TpExecution* tp)
    : ctx_(ctx), parameters_(weights), config_(weights.model.config().text), work_(work), kv_(kv),
      mtp_kv_(mtp_kv), state_(state), io_(io), prefill_hidden_(prefill_hidden),
      prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base), batch_text_kv_(batch_text_kv),
      batch_mtp_kv_(batch_mtp_kv), tp_(tp) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    embed_      = &parameters_.text.token_embedding;
    final_norm_ = &parameters_.text.final_norm;
    lm_head_    = &parameters_.text.output_head;
    mtp_        = parameters_.mtp ? &*parameters_.mtp : nullptr;
    if (mtp_enabled() && mtp_ == nullptr) {
        throw std::invalid_argument("MTP state requires selected MTP parameters");
    }
    if (parameters_.proposal) {
        const auto& p = *parameters_.proposal;
        set_proposal_head(
            &p.head, p.token_ids ? static_cast<const std::int32_t*>(p.token_ids->data) : nullptr,
            dimension(p.rows));
    }
    if (tp_ != nullptr) { validate_tensor_parallel(); }
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), T});

    auto roots = workspace::mtp_stem(work_, config_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 ||
            input_embeddings->ne[0] != dimension(config_.hidden_size) ||
            input_embeddings->numel() !=
                static_cast<std::int64_t>(dimension(config_.hidden_size)) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({dimension(config_.hidden_size), T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, mtp_->embedding_norm, config_.rms_norm_eps, true, e, s);
    ops::rmsnorm(flat_hidden, mtp_->hidden_norm, config_.rms_norm_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    project(fc_in, mtp_->input_projection, x, work_, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, mtp_->input_norm, config_.rms_norm_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace::mtp_attention_projection(work_, config_, T);
    Tensor q              = projection.query.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_attention_heads), T});
    Tensor k              = projection.key.view({dimension(config_.attention->head_dim),
                                                 dimension(config_.attention->num_key_value_heads), T});
    Tensor gate           = projection.gate.view({dimension(config_.attention->head_dim),
                                                  dimension(config_.attention->num_attention_heads), T});
    Tensor v              = projection.value.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat         = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat      = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat         = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat         = v.view({dimension(config_.attention->key_width()), T});
    mtp_projection(ah, mtp_->projection, *config_.attention, q_flat, gate_flat, k_flat, v_flat,
                   work_, s);

    const auto results = workspace::mtp_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, &parameters_.rope_scaling, qn, kn, ctx_.execution_view());

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
            *active_backend_kv_table_rows_,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, ctx_.execution_view());
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, ctx_.execution_view());
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace::mtp_post_attention(work_, config_, T);
    Tensor o        = post.output;
    project(a.view({dimension(config_.attention->query_width()), T}), mtp_->output, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        ffn(mh, mtp_->ffn, x, {}, work_, ctx_.execution_view(), true);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({dimension(config_.hidden_size), T});
    ops::rmsnorm(x, mtp_->final_norm, config_.rms_norm_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ah_last = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        Tensor v_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        mtp_kv_projection(ah, mtp_->projection, *config_.attention, k_flat, v_flat, work_, s);
        Tensor k = k_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor v = v_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor kn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_key_value_heads), T});
        ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
        text_rope(rope_positions, *config_.rope_parameters, &parameters_.rope_scaling, kn, ctx_.execution_view());
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(dimension(config_.hidden_size)) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat = work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        Tensor gate_flat =
            work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        mtp_query_gate_projection(ah_last, mtp_->projection, *config_.attention, q_flat, gate_flat,
                                  work_, s);
        Tensor q    = q_flat.view({dimension(config_.attention->head_dim),
                                   dimension(config_.attention->num_attention_heads), 1});
        Tensor gate = gate_flat.view({dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        Tensor qn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        text_rope(last_rope_position, *config_.rope_parameters, &parameters_.rope_scaling, qn, ctx_.execution_view());

        Tensor a = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), 1});
        ops::causal_softmax_attention_cached(
            qn, last_position,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            mtp_kv_.layer_view(0), envelope, work_, a, ctx_.execution_view());
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        project(a.view({dimension(config_.attention->query_width()), 1}), mtp_->output, o, work_,
                s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ops::rmsnorm(x_last, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            ffn(mh, mtp_->ffn, x_last, {}, work_, ctx_.execution_view(), true);
        }
        ops::rmsnorm(x_last, mtp_->final_norm, config_.rms_norm_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    auto proposal_scope = work_.scope();
    const int T         = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, dimension(config_.vocab_size), T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens,
                    proposal_head_ids_
                        ? proposal_head_n_
                        : dimension(parameters_.model.resources().public_token_count),
                    ctx_.stream);
        if (proposal_head_ids_ != nullptr) {
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = matrix_window(logits, T);
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    require_single_device("MTP forward");
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    require_single_device("MTP proposal");
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "ordinary decode logits");
    if (tp2()) {
        ordinary_decode_batch_tp2(ids, cache_positions, rope_positions, kv_table_rows,
                                  linear_state_source_slots, linear_state_destination_slots,
                                  envelope, hidden, logits);
        return;
    }

    cudaStream_t stream = ctx_.stream;
    work_.reset();
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

        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden, stream);
        project(hidden, *lm_head_, logits, work_, stream);
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           Tap& tap) {
    require_single_device("target verification");
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), columns});
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, flat_hidden, stream);
        project(flat_hidden, *lm_head_, flat_logits, work_, stream);
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    require_single_device("MTP decode");
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    require_single_device("MTP proposal");
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const BlockParameters& w, Tensor& x, int fidx, Phase ph) {
    const auto& p  = std::get<AttentionParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace::text_attention_projection(work_, config_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, w.input_norm, config_.rms_norm_eps, true, h, s);

    Tensor q         = projection.query.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_attention_heads), T});
    Tensor gate      = projection.gate.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), T});
    Tensor k         = projection.key.view({dimension(config_.attention->head_dim),
                                            dimension(config_.attention->num_key_value_heads), T});
    Tensor v         = projection.value.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat    = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat    = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat    = v.view({dimension(config_.attention->key_width()), T});
    attention_projection(h, p, q_flat, gate_flat, k_flat, v_flat, work_, s);

    const auto results = workspace::text_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, p.query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, p.key_norm, config_.rms_norm_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, &parameters_.rope_scaling, qn, kn, ctx_.execution_view());

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_,
            a_batch, ctx_.execution_view());
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, cache_positions, Tensor{}, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_, a,
            ctx_.execution_view());
    }
    ops::sigmoid_mul(gate, a, s);

    ops::linear_add(a.view({dimension(config_.attention->query_width()), T}), p.output.weight, x,
                    p.output.policy, work_, s);
}

void TextContext::gdn_mix(const BlockParameters& w, Tensor& x, int gidx, Phase ph) {
    const auto& p  = std::get<GdnParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace::gdn_control(work_, config_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    gdn_norm_control(x, w.input_norm, config_.rms_norm_eps, p, h, g, beta, work_,
                     ctx_.execution_view());

    const auto projection = workspace::gdn_projection(work_, config_, T);
    Tensor z  = projection.output_gate.view({dimension(config_.gdn->linear_value_head_dim),
                                             dimension(config_.gdn->linear_num_value_heads), T});
    Tensor qc = projection.query;
    Tensor kc = projection.key;
    Tensor vc = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        Tensor projection_input =
            h.view({dimension(config_.hidden_size), width, active_sequence_batch_});
        Tensor query_output =
            qc.view({dimension(config_.gdn->key_width()), width, active_sequence_batch_});
        Tensor key_output =
            kc.view({dimension(config_.gdn->key_width()), width, active_sequence_batch_});
        Tensor value_output =
            vc.view({dimension(config_.gdn->value_width()), width, active_sequence_batch_});
        Tensor gate_output =
            z.view({dimension(config_.gdn->value_width()), width, active_sequence_batch_});
        Tensor conv_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            gdn_projection_record(projection_input, p, *config_.gdn, conv_states, valid,
                                  *active_linear_state_source_slots_, records.conv, query_output,
                                  key_output, value_output, gate_output, work_, s);
        } else {
            gdn_projection_snapshot(projection_input, p, *config_.gdn, conv_states, valid,
                                    *active_linear_state_source_slots_,
                                    *active_linear_state_destination_slots_, query_output,
                                    key_output, value_output, gate_output, work_, s);
        }
    } else {
        Tensor qkv    = workspace::gdn_prefill_conv(work_, config_, T);
        Tensor z_flat = z.view({dimension(config_.gdn->value_width()), T});
        gdn_projection(h, p, qkv, z_flat, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc, kc, vc,
                                      s);
    }

    Tensor q_recurrent = qc.view({dimension(config_.gdn->linear_key_head_dim),
                                  dimension(config_.gdn->linear_num_key_heads), T});
    Tensor k_recurrent = kc.view({dimension(config_.gdn->linear_key_head_dim),
                                  dimension(config_.gdn->linear_num_key_heads), T});

    Tensor vv = vc.view({dimension(config_.gdn->linear_value_head_dim),
                         dimension(config_.gdn->linear_num_value_heads), T});
    Tensor o  = workspace::gdn_recurrent_output(work_, config_, T)
                   .view({dimension(config_.gdn->linear_value_head_dim),
                          dimension(config_.gdn->linear_num_value_heads), T});
    if (ph == Phase::Verify) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch           = q_recurrent.view({dimension(config_.gdn->linear_key_head_dim),
                                                     dimension(config_.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor k_batch           = k_recurrent.view({dimension(config_.gdn->linear_key_head_dim),
                                                     dimension(config_.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor v_batch           = vv.view({dimension(config_.gdn->linear_value_head_dim),
                                            dimension(config_.gdn->linear_num_value_heads), width,
                                            active_sequence_batch_});
        Tensor g_batch =
            g.view({dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor beta_batch = beta.view(
            {dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor out_batch =
            o.view({dimension(config_.gdn->linear_value_head_dim),
                    dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                recurrent_states, valid, *active_linear_state_source_slots_, records.key,
                records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta,
                             static_cast<float>(1.0 / std::sqrt(static_cast<double>(
                                                          config_.gdn->linear_key_head_dim))),
                             /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out,
                             o, ctx_.execution_view());
    }

    Tensor on = workspace::gdn_normalized_output(work_, config_, T)
                    .view({dimension(config_.gdn->linear_value_head_dim),
                           dimension(config_.gdn->linear_num_value_heads), T});
    ops::gated_rmsnorm(o, p.norm, z, config_.rms_norm_eps, on, ctx_.execution_view());

    ops::linear_add(on.view({dimension(config_.gdn->value_width()), T}), p.output.weight, x,
                    p.output.policy, work_, s);
}

ops::SparseMoeHints TextContext::next_projection_hints(int layer) const {
    const auto next = static_cast<std::size_t>(layer) + 1;
    return next < parameters_.text.layers.size() ? parameters_.text.layers[next].projection_prefetch
                                                 : ops::SparseMoeHints{};
}

void TextContext::mlp_tail(const BlockParameters& weights, Tensor& x, Phase,
                           const ops::SparseMoeHints& hints) {
    Tensor h = workspace::post_mixer_hidden(work_, config_, x.ne[1]);
    ops::rmsnorm(x, weights.post_attention_norm, config_.rms_norm_eps, true, h, ctx_.stream);
    ffn(h, weights.ffn, x, hints, work_, ctx_.execution_view());
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block  = parameters_.text.layers[layer];
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
                auto scope = work_.scope();
                if (full) {
                    attn_mix(block, x, compact, ph);
                } else {
                    gdn_mix(block, x, compact, ph);
                }
            }
            {
                nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer
                                                : nvtx::Name::VerifyPostMixer,
                                        nvtx::Category::PostMixer, layer);
                auto scope = work_.scope();
                mlp_tail(block, x, ph, next_projection_hints(static_cast<int>(layer)));
            }
            if constexpr (Tap::enabled) {
                tap.capture_layer(static_cast<int>(layer), x, ctx_.stream);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                     (prefill ? " prefill" : " verify") +
                                     " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
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
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
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
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        // Vision has its own interval. The Program-owned pair measures this text chunk,
        // including MTP alignment or the draft feature consumer submitted below.
        if (prefill_gpu_timer_) { prefill_gpu_timer_->start(); }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace::text_prefill_roots(
                work_, config_, len, rope_axes,
                static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
            ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                project(last_xf, *lm_head_, logits, work_, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
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
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

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
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden =
                            work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {dimension(config_.hidden_size), 1},
                                     "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    if (prefill_gpu_timer_) { prefill_gpu_timer_->record_stop(); }
    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    if (prefill_gpu_timer_) {
        timing.include({.gpu_elapsed_ns = static_cast<std::uint64_t>(
                            static_cast<double>(prefill_gpu_timer_->elapsed_ms()) * 1.0e6 + 0.5)});
    }
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

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
                view.storage != KvCacheStorage::Int8Group64) {
                throw std::invalid_argument(
                    "tensor-parallel attention supports only BF16 and INT8-G64 KV caches");
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
            view.storage != KvCacheStorage::Int8Group64) {
            throw std::invalid_argument(
                "tensor-parallel MTP attention supports only BF16 and INT8-G64 KV caches");
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
        text_rope(rope_for_op, *config_.rope_parameters, &parameters_.rope_scaling, qn[r], kn[r],
                    rank_execution_view(rank));

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
                                          *active_causal_attention_envelope_, *ws[r], a_batch,
                                          rank_execution_view(rank));
        } else {
            ops::causal_softmax_attention(qn[r], kn[r], v, cache_positions, Tensor{}, kv_table_rows,
                                          {head_dim, q_heads, kv_heads}, scale, cache,
                                          *active_causal_attention_envelope_, *ws[r], a[r],
                                          rank_execution_view(rank));
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
        ops::gated_rmsnorm(o[r], p[r]->norm, gate, config_.rms_norm_eps, on[r],
                       rank_execution_view(rank));
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
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "text/layers/" + std::to_string(layer) + (prefill ? " prefill" : " verify") +
                " tp2 columns=" + std::to_string(x[0].ne[1]) + ": " + error.what());
        }
    }
}

void TextContext::logits_tp2(const RankTensors& hidden, Tensor& logits) {
    logits_tp2(hidden, {&parameters_.text.output_head, &tp_->parameters->text.output_head}, logits,
               nullptr);
}

void TextContext::logits_tp2(const RankTensors& hidden,
                             const std::array<const LinearParameters*, 2>& head, Tensor& logits,
                             const Tensor* peer_logits) {
    const auto ws              = workspaces();
    const std::int32_t columns = hidden[0].ne[1];
    auto scope0                = work_.scope();
    auto scope1                = tp_->work->scope();
    const auto part0 = workspace::tp_logits(*ws[0], config_, head[0]->weight.n, columns, false);
    const auto part1 =
        workspace::tp_logits(*ws[1], config_, head[1]->weight.n, columns, peer_logits == nullptr);
    output_logits_split(hidden, head, {part0.partial, part1.partial},
                        {logits, peer_logits != nullptr ? *peer_logits : part1.gathered}, ws,
                        *tp_->execution, *tp_->events);
}

template <class Tap>
PrefillChunkResult TextContext::prefill_impl_tp2(std::span<const int> ids,
                                                 const TextPrefill& text_prefill, Tap& tap,
                                                 bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    const std::uint32_t base = text_kv_base_;
    if (base != text_prefill.begin ||
        text_prefill.token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
        throw std::invalid_argument("text prefill chunk does not match its full prompt");
    }
    const ExecutionContext& execution = *tp_->execution;
    const DeviceScope device(ctx_.device);
    cudaStream_t s  = ctx_.stream;
    const int T     = static_cast<int>(ids.size());
    const int chunk = static_cast<int>(prefill_chunk_);
    if (text_kv_base_ == 0) { rope_delta_ = 0; }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);
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
    const bool is_last = finalize_at_end && len == T;
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    const auto ws = workspaces();
    work_.reset();
    tp_->work->reset();
    {
        // A text chunk continuing a multimodal prefix keeps that prefix's RoPE delta, as on one
        // device; each rank offsets its own positions with its own copy of the delta.
        const std::int32_t rope_axes = rope_delta_ != 0 ? 1 : 0;
        std::array<workspace::TextPrefillRoots, 2> roots;
        RankTensors staging;
        for (std::size_t r = 0; r < 2; ++r) {
            roots[r]   = workspace::text_prefill_roots(*ws[r], config_, len, rope_axes, 0);
            staging[r] = workspace::tp_call_roots(*ws[r], config_, len).staging;
        }
        Tensor peer_rope_delta = rope_delta_ != 0 ? ws[1]->alloc(DType::I32, {1}) : Tensor{};
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
            if (rope_delta_ != 0) {
                Tensor delta = io_.rope_delta;
                if (rank != 0) {
                    delta = peer_rope_delta;
                    ops::set_i32_scalar(delta, rope_delta_, stream);
                }
                rope_positions[r] = roots[r].rope_positions;
                ops::offset_i32_positions(positions[r], delta, rope_positions[r], stream);
            }
            x[r] = roots[r].residual;
            ops::embedding(roots[r].ids, rank_parameters(rank).text.token_embedding, x[r], stream);
        });

        ScopedPositions scoped_cache(active_cache_positions_, positions[0]);
        ScopedPositions scoped_rope(active_rope_positions_, rope_positions[0]);
        ScopedValue<const Tensor*> peer_cache(peer_cache_positions_, &positions[1]);
        ScopedValue<const Tensor*> peer_rope(peer_rope_positions_, &rope_positions[1]);
        ScopedValue<const Tensor*> peer_rows(peer_kv_table_rows_, &tp_->text_kv_table_row);
        const auto visible = static_cast<std::uint32_t>(base_i + len);
        const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
        ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

        if constexpr (Tap::enabled) { tap.begin(x[0]); }
        run_layers_tp2(x, Phase::Prefill, staging, tap);
        if constexpr (requires { tap.capture_positions(positions[0], s); }) {
            tap.capture_positions(positions[0], s);
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
            const DeviceScope peer(execution.dev[1]->device);
            ops::rmsnorm(x[1], rank_parameters(1).text.final_norm, config_.rms_norm_eps, true,
                         peer_xf, rank_stream(1));
        }

        if (is_last) {
            Tensor peer_last;
            if (prepare_mtp_prompt) {
                peer_last = peer_xf.slice(1, len - 1, 1);
            } else {
                peer_last = ws[1]->alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                const DeviceScope peer(execution.dev[1]->device);
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
            const auto alignment_tokens = static_cast<std::uint32_t>(text_prefill.token_ids.size());
            const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                alignment_tokens, base, static_cast<std::uint32_t>(len));
            const int prompt_columns =
                len - static_cast<int>(mtp_window.final_column_uses_generated_token);
            Tensor mtp_ids = work_.alloc(DType::I32, {len});
            if (prompt_columns != 0) {
                Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                copy_i32(text_prefill.token_ids.data() + mtp_window.shifted_embedding_begin,
                         prompt_mtp_ids, s);
            }
            if (mtp_window.final_column_uses_generated_token) {
                Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
            const RankTensors mtp_input{xf, peer_xf};
            if (is_last && mtp_proposal_extent_ != 0) {
                Tensor logits = matrix_window(io_.logits, 1);
                Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                const RankTensors ar_hidden{io_.mtp->ar_hidden, tp_->mtp->ar_hidden};
                mtp_prefill_chunk_tp2(mtp_ids, mtp_input, positions, rope_positions, chunk_envelope,
                                      true, &ar_hidden, &logits, &draft0);

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
                mtp_prefill_chunk_tp2(mtp_ids, mtp_input, positions, rope_positions, chunk_envelope,
                                      false, nullptr, nullptr, nullptr);
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
                const DeviceScope peer(execution.dev[1]->device);
                CUDA_CHECK(cudaMemcpyAsync(
                    peer_rewrite_checkpoint_hidden_output_->data, peer_checkpoint_hidden.data,
                    peer_checkpoint_hidden.bytes(), cudaMemcpyDeviceToDevice, rank_stream(1)));
            }
        }
    }

    if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
        work_.reset();
        tap.consume_prefill_chunk(len, split_rel > 0 && len == split_rel);
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

void TextContext::target_verify_batch(
    const RankTensors& ids, const RankTensors& cache_positions, const RankTensors& rope_positions,
    const RankTensors& valid_columns, const RankTensors& kv_table_rows,
    const RankTensors& linear_state_source_slots, ops::CausalAttentionExecutionEnvelope envelope,
    const RankTensors& hidden, const RankTensors& logits, Tensor& target_tokens) {
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
    // out-of-bounds write, so both ranks' extents are checked.
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
        require_tensor_shape(logits[r], DType::BF16, {V, width, batch},
                             "target verify batch logits");
    }
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    const ExecutionContext& execution = *tp_->execution;
    const DeviceScope device(ctx_.device);
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
        NullTap tap;
        run_layers_tp2(x, Phase::Verify, staging, tap);

        RankTensors flat_hidden;
        for (std::size_t r = 0; r < 2; ++r) { flat_hidden[r] = hidden[r].view({H, columns}); }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], rank_parameters(rank).text.final_norm, config_.rms_norm_eps, true,
                         flat_hidden[r], rank_stream(rank));
        });
        Tensor flat_logits       = logits[0].view({V, columns});
        const Tensor peer_logits = logits[1].view({V, columns});
        logits_tp2(flat_hidden, {&parameters_.text.output_head, &tp_->parameters->text.output_head},
                   flat_logits, &peer_logits);
        // Acceptance is rank 0's alone, so only rank 0 needs the target tokens.
        Tensor flat_tokens = target_tokens.view({columns});
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
    work_.reset();
    tp_->work->reset();
}

void TextContext::mtp_forward_stem_tp2(const Tensor& ids, const RankTensors& hidden, RankTensors& x,
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
    // Only rank 0 embeds tokens; the plan sizes both ranks for rank 0's roots.
    std::array<workspace::MtpStemSplitRoots, 2> roots{
        workspace::mtp_stem_split(*ws[0], config_, T, true),
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
            Tensor embedding = roots[0].embedding;
            ops::embedding(flat_ids, *embed_, embedding, stream);
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
        text_rope(rope_for_op, *config_.rope_parameters, &parameters_.rope_scaling, qn[r], kn[r],
                  rank_execution_view(rank));

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
                {head_dim, q_heads, kv_heads}, scale, cache, envelope, *ws[r], a_batch,
                rank_execution_view(rank));
        } else {
            ops::causal_softmax_attention(qn[r], kn[r], v, positions[r], Tensor{}, rows,
                                          {head_dim, q_heads, kv_heads}, scale, cache, envelope,
                                          *ws[r], a[r], rank_execution_view(rank));
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
                                       const RankTensors& mtp_hidden) {
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
    mtp_forward_stem_tp2(ids, hidden, x, ah, staging);
    mtp_forward_tail_tp2(x, ah, positions, rope_positions, envelope, mtp_hidden, staging);
}

void TextContext::mtp_prefill_chunk_tp2(const Tensor& ids, const RankTensors& hidden,
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
            rope_positions[r].ne[1] != 1 || rope_positions[r].ne[2] != 1 ||
            rope_positions[r].ne[3] != 1 || !rope_positions[r].is_contiguous() ||
            rope_positions[r].data == nullptr) {
            throw std::invalid_argument("tensor-parallel MTP prefill rope positions must be [T]");
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
        mtp_forward_stem_tp2(ids, hidden, x, ah, staging);

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
            text_rope(rope_positions[r], *config_.rope_parameters, &parameters_.rope_scaling, kn[r],
                      rank_execution_view(rank));
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
    for (std::size_t r = 0; r < 2; ++r) {
        qn[r] = ws[r]->alloc(DType::BF16, {head_dim, q_heads, 1});
        a[r]  = ws[r]->alloc(DType::BF16, {head_dim, q_heads, 1});
        o[r]  = ws[r]->alloc(DType::BF16, {H, 1});
        mh[r] = ws[r]->alloc(DType::BF16, {H, 1});
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t stream = rank_stream(rank);
        Tensor q            = q_flat[r].view({head_dim, q_heads, 1});
        Tensor gate         = gate_flat[r].view({head_dim, q_heads, 1});
        ops::rmsnorm(q, rank_mtp(rank).query_norm, config_.rms_norm_eps, true, qn[r], stream);
        const Tensor last_position = positions[r].slice(0, T - 1, 1);
        const Tensor last_rope     = rope_positions[r].slice(0, T - 1, 1);
        text_rope(last_rope, *config_.rope_parameters, &parameters_.rope_scaling, qn[r],
                  rank_execution_view(rank));
        ops::causal_softmax_attention_cached(
            qn[r], last_position, {head_dim, q_heads, kv_heads}, scale,
            rank_mtp_kv(rank).layer_view(0), envelope, *ws[r], a[r], rank_execution_view(rank));
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
    proposal_argmax_tp2(*final_hidden, *logits, nullptr, *draft_token);
}

void TextContext::proposal_argmax_tp2(const RankTensors& hidden, Tensor& logits,
                                      const Tensor* peer_logits, Tensor& proposal_tokens) {
    const std::int32_t T = hidden[0].ne[1];
    const std::int32_t H = dimension(config_.hidden_size);
    const std::int32_t V = dimension(config_.vocab_size);
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {H, T}, "proposal hidden");
    }
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, V, T, "proposal logits");
    if (proposal_head_ != nullptr) {
        // The optimized proposal head is PrimaryOnly: rank 0 projects, picks and maps alone.
        proposal_argmax(hidden[0], logits, proposal_tokens);
        return;
    }
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    Tensor output_logits = matrix_window(logits, T);
    Tensor peer_window;
    if (peer_logits != nullptr) {
        Tensor peer = *peer_logits;
        require_tensor_window(peer, DType::BF16, V, T, "proposal peer logits");
        peer_window = matrix_window(peer, T);
    }
    // The full head is the vocabulary-split output head: the argmax runs over the gathered
    // complete logits, on rank 0.
    logits_tp2(hidden, {&rank_mtp(0).output_head, &rank_mtp(1).output_head}, output_logits,
               peer_logits != nullptr ? &peer_window : nullptr);
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
    const DeviceScope device(ctx_.device);
    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows[0]);
    ScopedValue<const Tensor*> peer_backend_binding(peer_backend_kv_table_rows_, &kv_table_rows[1]);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
    ScopedValue<const Tensor*> peer_valid_binding(peer_valid_columns_, &valid_columns[1]);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core_tp2(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_propose_batch(const RankTensors& hidden, const RankTensors& logits,
                                    Tensor& draft_tokens) {
    if (!tp2()) {
        throw std::invalid_argument("rank-pair MTP proposal requires tensor parallelism");
    }
    const std::int32_t batch = hidden[0].ne[1];
    for (std::size_t r = 0; r < 2; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {dimension(config_.hidden_size), batch},
                             "MTP proposal batch hidden");
        require_tensor_shape(logits[r], DType::BF16, {dimension(config_.vocab_size), batch},
                             "MTP proposal batch logits");
    }
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    const DeviceScope device(ctx_.device);
    auto proposal_scope0 = work_.scope();
    auto proposal_scope1 = tp_->work->scope();
    Tensor output_logits = logits[0];
    proposal_argmax_tp2(hidden, output_logits, &logits[1], draft_tokens);
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
    const DeviceScope device(ctx_.device);
    const auto ws        = workspaces();
    auto position_scope0 = work_.scope();
    auto position_scope1 = tp_->work->scope();
    RankTensors rope_position;
    for (std::size_t r = 0; r < 2; ++r) { rope_position[r] = ws[r]->alloc(DType::I32, {1}); }
    // Each rank offsets its own copy of the position by its own copy of the RoPE delta.
    Tensor peer_delta = ws[1]->alloc(DType::I32, {1});
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t stream = rank_stream(rank);
        Tensor delta        = io_.rope_delta;
        if (rank != 0) {
            delta = peer_delta;
            ops::set_i32_scalar(delta, rope_delta_, stream);
        }
        ops::offset_i32_positions(position[r], delta, rope_position[r], stream);
    });
    mtp_forward_core_tp2(token, previous_hidden, position, rope_position, envelope, mtp_hidden);
    auto logits_scope0 = work_.scope();
    auto logits_scope1 = tp_->work->scope();
    proposal_argmax_tp2(mtp_hidden, logits, nullptr, draft_token);
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
    const DeviceScope device(ctx_.device);
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
            const DeviceScope peer(execution.dev[1]->device);
            ops::rmsnorm(x[1], rank_parameters(1).text.final_norm, config_.rms_norm_eps, true,
                         peer_hidden, rank_stream(1));
        }
        logits_tp2({hidden, peer_hidden}, logits);
    }
    work_.reset();
    tp_->work->reset();
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    if (tp2()) {
        return prefill_impl_tp2(full_ids.subspan(begin, nominal_length), text_prefill, tap,
                                finalize_at_end);
    }
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    if (tp2()) {
        return prefill_impl_tp2(full_ids.subspan(begin, nominal_length), text_prefill, sink,
                                finalize_at_end);
    }
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    require_single_device("multimodal prefill");
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    require_single_device("multimodal prefill");
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, sink,
                        finalize_at_end);
}

} // namespace ninfer::models::qwen3_5::execution
