#pragma once
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/execution/tp.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using Phase = qwen3_5::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

// One Text call over the rank-0 operands, and with a non-null `tp` over both ranks of a
// two-device Model (tp.h). A null `tp` is the single-device schedule, unchanged. With `tp`, the
// text prefill chunk (with its MTP prompt alignment or DFlash feature capture), ordinary decode,
// speculative target verification (with or without DFlash feature capture), the MTP decode-round
// forwards and proposals, the MTP bridge and their logits run split through the rank-array
// overloads below; the single-rank MTP and verification entries and multimodal prefill are
// rejected with std::invalid_argument. The constructor rejects the MoE FFN, paired
// input projections, KV caches the head-local attention does not support (only BF16 and INT8-G64)
// and incomplete MTP bindings.
class TextContext {
public:
    // Rank 0's operand then rank 1's, each resident on that rank's device.
    using RankTensors = std::array<Tensor, kTensorParallelWidth>;

    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr,
                const TpExecution* tp                      = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_rope_delta(std::int32_t delta) noexcept { rope_delta_ = delta; }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_prefill_gpu_timer(CudaEventTimer* timer) noexcept { prefill_gpu_timer_ = timer; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    // `peer_output` is rank 1's StateImage column at tensor-parallel width 2 under MTP, the only
    // backend whose rank 1 reads a retained hidden (the MTP bridge); null otherwise.
    void set_rewrite_checkpoint_hidden_output(Tensor* output,
                                              Tensor* peer_output = nullptr) noexcept {
        rewrite_checkpoint_hidden_output_      = output;
        peer_rewrite_checkpoint_hidden_output_ = peer_output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);

    // --- tp == 2 speculative forms --------------------------------------------------------------
    // Rank 1's control tensors are its own copies of rank 0's values (the Program uploads the same
    // host ingress record to both ranks and derives the rest with the same Ops). Token ids an MTP
    // call embeds are rank 0's alone: rank 0's half of the MTP input projection contracts the
    // token embedding and rank 1's the target hidden. Sampling, acceptance and the winning draft
    // ids are rank 0's, and so are the logits: rank 0 alone gathers them, into its own frames.

    // Verification with ReplaySSM records (GdnStateAction::RecordForReplay) on each rank. Both
    // ranks keep their normalized hidden [H,W,B]; rank 0's `logits` receive the complete [V,W,B]
    // logits and rank 0 writes the per-column argmax into `target_tokens`.
    void target_verify_batch(const RankTensors& ids, const RankTensors& cache_positions,
                             const RankTensors& rope_positions, const RankTensors& valid_columns,
                             const RankTensors& kv_table_rows,
                             const RankTensors& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope,
                             const RankTensors& hidden, Tensor& logits, Tensor& target_tokens);
    // The same verification with rank 0's DFlash drafter capturing its target features: `sink`
    // reads rank 0's residual after each captured layer's all-reduce (the complete hidden state)
    // on rank 0's stream. Rank 1 captures nothing.
    void target_verify_batch(const RankTensors& ids, const RankTensors& cache_positions,
                             const RankTensors& rope_positions, const RankTensors& valid_columns,
                             const RankTensors& kv_table_rows,
                             const RankTensors& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope,
                             const RankTensors& hidden, Tensor& logits, Tensor& target_tokens,
                             DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const RankTensors& hidden,
                                  const RankTensors& cache_positions,
                                  const RankTensors& rope_positions,
                                  const RankTensors& valid_columns,
                                  const RankTensors& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  const RankTensors& mtp_hidden);
    // `logits` [V,B] is rank 0's.
    void mtp_propose_batch(const RankTensors& hidden, Tensor& logits, Tensor& draft_tokens);
    // The MTP bridge over both ranks: `ids` [T] (rank 0's alone) against each rank's copy of the
    // target hidden [H,T], appending both ranks' MTP K/V at `positions` through each rank's
    // prefill MTP KV row. `rope_positions[r]` are one-axis [T]. With `logits_column` >= 0 rank 0
    // proposes `draft_token` from that column of `mtp_hidden`, over logits rank 0 alone gathers.
    void mtp_forward_batch(const Tensor& ids, const RankTensors& hidden,
                           const RankTensors& positions, const RankTensors& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           const RankTensors& mtp_hidden, int logits_column, Tensor* logits,
                           Tensor* draft_token);
    // One prompt proposal step through each rank's prefill MTP KV row; rank 0 alone gathers the
    // proposal logits.
    void mtp_forward_ar_step(const Tensor& token, const RankTensors& previous_hidden,
                             const RankTensors& position,
                             ops::CausalAttentionExecutionEnvelope envelope,
                             const RankTensors& mtp_hidden, Tensor& logits, Tensor& draft_token);

private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    // --- tp == 2 -------------------------------------------------------------------------------
    // Separate functions rather than branches inside the single-device ones, so the tp1 schedule
    // is unchanged. The residual x[r] is replicated: each layer ends in an all-reduce that leaves
    // the identical BF16 sum on both ranks, so every later per-rank stage sees identical inputs.
    // Rank r's work is issued on ExecutionContext::dev[r]'s stream; only the collectives inside
    // the row-parallel projections order the two streams, with no host synchronization.
    using RankBlocks = std::array<const BlockParameters*, kTensorParallelWidth>;

    [[nodiscard]] bool tp2() const noexcept { return tp_ != nullptr; }

    void validate_tensor_parallel();
    void require_single_device(const char* operation) const;
    [[nodiscard]] const Parameters& rank_parameters(int rank) const noexcept;
    [[nodiscard]] std::array<WorkspaceArena*, kTensorParallelWidth> workspaces() const noexcept;
    [[nodiscard]] cudaStream_t rank_stream(int rank) const noexcept;
    // The per-rank execution view: the rank's own stream and SM count. Ops that select launch
    // shapes from the device take this, not a bare stream, so rank 1 never inherits rank 0's
    // device properties.
    [[nodiscard]] DeviceExecutionView rank_execution_view(int rank) const noexcept;
    [[nodiscard]] LinearAttentionStatePool& rank_state(int rank) const noexcept;
    [[nodiscard]] const qwen3_5::PagedKVCache& rank_text_cache(int rank) const;
    // Rank 0 reads the active_* bindings exactly as the single-device path does; rank 1 reads
    // the peer_* bindings the tp2 entry point sets from rank 1's own control tensors.
    [[nodiscard]] const Tensor& rank_cache_positions(int rank) const;
    [[nodiscard]] const Tensor& rank_rope_positions(int rank) const;
    [[nodiscard]] const Tensor& rank_kv_table_rows(int rank) const;
    [[nodiscard]] const Tensor& rank_linear_state_source_slots(int rank) const;
    [[nodiscard]] const Tensor& rank_linear_state_destination_slots(int rank) const;
    // Both ranks bind valid columns, or neither does (an empty Tensor: every column is live); two
    // ranks masking different columns would diverge silently.
    [[nodiscard]] Tensor rank_valid_columns(int rank) const;
    // The MTP execution rows: the batch binding of an MTP decode call, else the prefill scalar.
    [[nodiscard]] const Tensor& rank_backend_kv_table_rows(int rank) const;
    [[nodiscard]] const qwen3_5::PagedKVCache& rank_mtp_cache(int rank) const;
    [[nodiscard]] qwen3_5::PagedKVCacheView rank_mtp_kv(int rank) const;
    [[nodiscard]] const MtpParameters& rank_mtp(int rank) const;
    [[nodiscard]] const GdnReplayRecords& rank_replay_records(int rank) const;
    void validate_tensor_parallel_mtp();
    void attn_mix_tp2(const RankBlocks& weights, RankTensors& x, int index,
                      const RankTensors& staging);
    void gdn_mix_tp2(const RankBlocks& weights, RankTensors& x, int index, Phase phase,
                     const RankTensors& staging);
    void mlp_tail_tp2(const RankBlocks& weights, RankTensors& x, const RankTensors& staging);
    template <class Tap>
    void run_layers_tp2(RankTensors& x, Phase phase, const RankTensors& staging, Tap& tap);
    template <class Tap>
    void target_verify_batch_tp2_impl(const RankTensors& ids, const RankTensors& cache_positions,
                                      const RankTensors& rope_positions,
                                      const RankTensors& valid_columns,
                                      const RankTensors& kv_table_rows,
                                      const RankTensors& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      const RankTensors& hidden, Tensor& logits,
                                      Tensor& target_tokens, Tap& tap);
    // Vocabulary-split head: each rank projects its half of the vocabulary from its final hidden
    // columns, and rank 0 alone gathers the complete [V, C] logits into its `logits` (one pull of
    // rank 1's half).
    void logits_tp2(const RankTensors& hidden, Tensor& logits);
    void logits_tp2(const RankTensors& hidden, const std::array<const LinearParameters*, 2>& head,
                    Tensor& logits);
    // MTP head over both ranks: three all-reduces (input projection, attention output, post-mixer
    // down projection) leave the replicated residual identical on the two ranks, as in the text
    // layers. The attention runs over each rank's half of the heads and its own MTP KV pages.
    void mtp_forward_stem_tp2(const Tensor& ids, const RankTensors& hidden, RankTensors& x,
                              RankTensors& ah, const RankTensors& staging);
    void mtp_forward_tail_tp2(RankTensors& x, const RankTensors& ah, const RankTensors& positions,
                              const RankTensors& rope_positions,
                              ops::CausalAttentionExecutionEnvelope envelope,
                              const RankTensors& mtp_hidden, const RankTensors& staging);
    void mtp_forward_core_tp2(const Tensor& ids, const RankTensors& hidden,
                              const RankTensors& positions, const RankTensors& rope_positions,
                              ops::CausalAttentionExecutionEnvelope envelope,
                              const RankTensors& mtp_hidden);
    void mtp_prefill_chunk_tp2(const Tensor& ids, const RankTensors& hidden,
                               const RankTensors& positions, const RankTensors& rope_positions,
                               ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                               const RankTensors* final_hidden, Tensor* logits,
                               Tensor* draft_token);
    // The optimized proposal head proposes on rank 0 alone when PrimaryOnly, or on both ranks when
    // split by vocabulary (execution::proposal_argmax_split, rank 0 combining the two argmaxes);
    // the full head is the vocabulary-split output head, gathered before rank 0's argmax.
    void proposal_argmax_tp2(const RankTensors& hidden, Tensor& logits, Tensor& proposal_tokens);
    void ordinary_decode_batch_tp2(const Tensor& ids, const Tensor& cache_positions,
                                   const Tensor& rope_positions, const Tensor& kv_table_rows,
                                   const Tensor& linear_state_source_slots,
                                   const Tensor& linear_state_destination_slots,
                                   ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                   Tensor& logits);

    void attn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                  const ops::SparseMoeHints& hints);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);
    // Text-only tp2 prefill of one chunk. A DFlash feature tap reads rank 0's replicated residual.
    template <class Tap>
    [[nodiscard]] PrefillChunkResult prefill_impl_tp2(std::span<const int> ids,
                                                      const TextPrefill& text_prefill, Tap& tap,
                                                      bool finalize_at_end);
    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    // Borrowed from Program; every measured chunk retires before this pair is reused.
    CudaEventTimer* prefill_gpu_timer_ = nullptr;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    GdnStateAction gdn_state_action_               = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_        = nullptr;
    std::int64_t prefill_split_frontier_           = -1;
    Tensor* rewrite_checkpoint_hidden_output_      = nullptr;
    Tensor* peer_rewrite_checkpoint_hidden_output_ = nullptr;
    std::uint32_t mtp_proposal_extent_             = 0;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    const MtpParameters* mtp_                   = nullptr;

    const TpExecution* tp_ = nullptr;
    // Each rank's share of config_ (shard_text_config), present only at tp == 2.
    std::optional<TextConfig> shard_config_;
    const Tensor* peer_cache_positions_                = nullptr;
    const Tensor* peer_rope_positions_                 = nullptr;
    const Tensor* peer_kv_table_rows_                  = nullptr;
    const Tensor* peer_linear_state_source_slots_      = nullptr;
    const Tensor* peer_linear_state_destination_slots_ = nullptr;
    const Tensor* peer_valid_columns_                  = nullptr;
    const Tensor* peer_backend_kv_table_rows_          = nullptr;
};

} // namespace ninfer::models::qwen3_5::execution
