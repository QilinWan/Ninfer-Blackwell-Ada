#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "models/qwen3_5/program/storage/draft_context.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_prefill.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::models::qwen3_5::execution {

using qwen3_5::PreparedPromptData;
using detail::DFlashPersistentState;
using qwen3_5::PromptModality;

struct ExecutionCore {
    DeviceContext& device;
    const execution::Parameters& parameters;
    WorkspaceArena& work;
    LinearAttentionStatePool& linear_attention;
    const GdnReplayRecords* replay_records;
    qwen3_5::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
    // Tensor-parallel width 2 only: rank 1's operands, which every Text call built from this core
    // (prefill, forced tokens, ordinary decode) drives in lockstep, and the bridge that enrolls
    // rank 1's stream in a decode-graph capture.
    const TpExecution* tp                     = nullptr;
    const DecodeGraphPeerBridge* graph_bridge = nullptr;
};

struct PrefillContext {
    ExecutionCore execution;
    qwen3_5::PagedKVCacheView text_kv;
    qwen3_5::PagedKVCacheView mtp_kv;
    const qwen3_5::PagedKVCache& text_cache;
    const qwen3_5::PagedKVCache* mtp_cache;
    DFlashPersistentState* dflash;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* rewrite_checkpoint_hidden;
    std::int32_t state_source_slot                             = 0;
    std::int32_t state_destination_slot                        = 0;
    std::uint32_t mtp_proposal_extent                          = 0;
    std::int32_t dflash_kv_table_row                           = 0;
    qwen3_5::DFlashPrefillIngress* dflash_prefill_host_ingress = nullptr;
    std::int32_t rope_delta                                    = 0;
    CudaEventTimer* prefill_gpu_timer                          = nullptr;
    // Tensor-parallel width 2 under MTP: rank 1's column of the StateImage that rank 0's
    // `rewrite_checkpoint_hidden` names (the same slot of rank 1's mirror pool). Appended at the
    // end of this tree's field order and assigned at the prefill sites after construction.
    Tensor* peer_rewrite_checkpoint_hidden = nullptr;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const qwen3_5::PagedKVCache& text_cache;
    qwen3_5::OrdinaryDecodeState& frame;
    const qwen3_5::OrdinaryDecodeIngress& host_ingress;
    qwen3_5::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // Tensor-parallel width 2 only: rank 1's frame, which receives the same host ingress record.
    qwen3_5::OrdinaryDecodeState* peer_frame = nullptr;
};

struct MtpBatchContext {
    ExecutionCore execution;
    const qwen3_5::PagedKVCache& text_cache;
    const qwen3_5::PagedKVCache& mtp_cache;
    qwen3_5::MtpDecodeState& frame;
    const qwen3_5::MtpDecodeIngress& host_ingress;
    qwen3_5::MtpDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // Tensor-parallel width 2 only: rank 1's frame, which receives the same host ingress record,
    // and rank 1's continuation hidden store.
    qwen3_5::MtpDecodeState* peer_frame    = nullptr;
    Tensor* peer_continuation_hidden_store = nullptr;
};

struct DFlashBatchContext {
    ExecutionCore execution;
    const qwen3_5::PagedKVCache& text_cache;
    DFlashPersistentState& dflash; // Rank 0's; rank 1 holds no drafter state.
    qwen3_5::DFlashDecodeState& frame;
    const qwen3_5::DFlashDecodeIngress& host_ingress;
    qwen3_5::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    std::span<TokenId> host_drafts;
    CudaCompletionEvent& drafts_ready;
    // Tensor-parallel width 2 only: rank 1's frame, which receives the same host ingress record
    // and rank 0's draft tokens, and rank 1's continuation hidden store. Appended at the end of
    // this tree's field order and assigned at the round sites.
    qwen3_5::DFlashDecodeState* peer_frame = nullptr;
    Tensor* peer_continuation_hidden_store = nullptr;
};

struct DFlashAppendContext {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

struct MtpCausalAttentionEnvelopes {
    ops::CausalAttentionExecutionEnvelope target_verify;
    ops::CausalAttentionExecutionEnvelope batch;
    std::array<ops::CausalAttentionExecutionEnvelope, kMaximumMtpDraftTokens - 1> ar;
};

struct DFlashEnvelopes {
    ops::SlidingWindowAttentionExecutionEnvelope local;
    ops::ContextAttentionExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

struct TargetVerifyFrameView {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor target_hidden;
    Tensor target_logits;
    Tensor target_tokens;
    Tensor drafts;
    Tensor current_extents;
    Tensor candidate_ids;
    Tensor proposal_q;
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor selected_hidden;
    const GdnReplayRecords* replay_records = nullptr;
    const ops::SamplingConfig* sampling    = nullptr;
    DFlashFeatureSink* feature_sink        = nullptr;
};

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent);
void target_verify_forward(ExecutionCore&, TextContext&, TargetVerifyFrameView,
                           ops::CausalAttentionExecutionEnvelope);
void target_accept(ExecutionCore&, Tensor& continuation_hidden_store, TargetVerifyFrameView);
// Tensor-parallel width 2: both ranks verify their halves of the model, each with its own frame
// Tensor-parallel width 2: both ranks verify their halves of the model, each with its own frame
// view and ReplaySSM records. Acceptance (greedy, or sparse over `frame`'s proposal distribution)
// runs on rank 0 alone over the gathered logits; its accepted-draft counts are copied to rank 1,
// and both ranks select and publish their accepted hidden. `peer.replay_records` are rank 1's.
// Only `frame` may carry a DFlash feature sink, which captures rank 0's replicated residual;
// `peer` carries no sink and no proposal distribution. This tree splits verify and accept into two
// entries, so the peer face is one extra parameter each.
void target_verify_forward(ExecutionCore&, TextContext&, TargetVerifyFrameView frame,
                           TargetVerifyFrameView peer, ops::CausalAttentionExecutionEnvelope);
void target_accept(ExecutionCore&, Tensor& continuation_hidden_store, TargetVerifyFrameView frame,
                   TargetVerifyFrameView peer, Tensor& peer_continuation_hidden_store);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(PrefillContext& state,
                                                    std::span<const TokenId> ids,
                                                    std::uint32_t nominal_length,
                                                    std::optional<std::uint32_t> split_frontier,
                                                    bool finalize_at_end);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> split_frontier, bool finalize_at_end);

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
// Resumes the MTP head at `position` from the retained target hidden of that position. At
// tensor-parallel width 2 `peer_previous_hidden` is rank 1's retained copy (the same StateImage
// slot of its mirror pool) and both ranks run the split head; it is null on one device.
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, const Tensor* peer_previous_hidden,
                            std::int32_t position, std::span<const std::int32_t> rope_position,
                            bool build_proposal, const Tensor* next_embedding = nullptr);
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable);

// Executes one exact-B MTP verification/alignment/proposal transaction. Each row may carry a
// different current and next proposal extent while the model traversal remains batched.
enum class SpeculativePhase { Forward, Finish };

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition, SpeculativePhase phase);
void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable,
                      SpeculativePhase phase);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::CausalAttentionExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition, SpeculativePhase phase);
void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::CausalAttentionExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable, SpeculativePhase phase);

} // namespace ninfer::models::qwen3_5::execution
