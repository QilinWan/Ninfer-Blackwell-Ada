#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/program_impl.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::models::qwen3_5::execution {

void target_verify_forward(ExecutionCore& execution, TextContext& card, TargetVerifyFrameView frame,
                           ops::CausalAttentionExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
}

void target_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                   TargetVerifyFrameView frame) {
    if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            {false}, execution.work, execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            execution.work, execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
}

// Tensor-parallel width 2: both ranks verify their halves of the model, each with its own frame
// view and ReplaySSM records. Acceptance runs on rank 0 alone over the gathered logits; its
// accepted-draft counts are copied to rank 1, and both ranks select and publish their accepted
// hidden. `peer.replay_records` are rank 1's; feature capture is not supported. This tree splits
// the donor's single entry into its two existing phases.
void target_verify_forward(ExecutionCore& execution, TextContext& card,
                           TargetVerifyFrameView frame, TargetVerifyFrameView peer,
                           ops::CausalAttentionExecutionEnvelope envelope) {
    if (execution.tp == nullptr) {
        throw std::logic_error("tensor-parallel target verify requires rank 1");
    }
    if (frame.replay_records == nullptr || peer.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    // The drafter, its proposals and acceptance are rank 0's alone.
    if (peer.feature_sink != nullptr || peer.proposal_q.data != nullptr) {
        throw std::logic_error(
            "tensor-parallel target verify captures features and accepts on rank 0 only");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    const TextContext::RankTensors hidden{frame.target_hidden, peer.target_hidden};
    const TextContext::RankTensors logits{frame.target_logits, peer.target_logits};
    if (frame.feature_sink != nullptr) {
        // Feature capture exists only on rank 0: it records rank 0's replicated residual.
        if (peer.feature_sink != nullptr || peer.proposal_q.data != nullptr) {
            throw std::logic_error(
                "tensor-parallel target verify requires the peer frame to carry no feature sink "
                "and no proposal distribution");
        }
        card.target_verify_batch(
            {frame.ids, peer.ids}, {frame.cache_positions, peer.cache_positions},
            {frame.rope_positions, peer.rope_positions}, {frame.valid_columns, peer.valid_columns},
            {frame.kv_table_rows, peer.kv_table_rows},
            {frame.state_source_slots, peer.state_source_slots}, envelope, hidden, logits,
            frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(
            {frame.ids, peer.ids}, {frame.cache_positions, peer.cache_positions},
            {frame.rope_positions, peer.rope_positions}, {frame.valid_columns, peer.valid_columns},
            {frame.kv_table_rows, peer.kv_table_rows},
            {frame.state_source_slots, peer.state_source_slots}, envelope, hidden, logits,
            frame.target_tokens);
    }
}

void target_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                   TargetVerifyFrameView frame, TargetVerifyFrameView peer,
                   Tensor& peer_continuation_hidden_store) {
    if (execution.tp == nullptr) {
        throw std::logic_error("tensor-parallel target accept requires rank 1");
    }
    // Acceptance runs on rank 0 alone: greedy drafts, or sparse drafts over rank 0's proposal
    // distribution (the vocabulary-split head gathers the full logits there).
    if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            {false}, execution.work, execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            execution.work, execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);

    // Rank 1 reads the accepted counts across devices once rank 0's acceptance is complete. The
    // copy is a pull on rank 1's stream over unified addressing, as the collectives' transfers.
    const DeviceContext& rank1    = *execution.tp->execution->dev[1];
    const ops::PeerEvents& events = *execution.tp->events;
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), execution.device.stream));
    const detail::ScopedCurrentDevice scope(rank1.device);
    CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, events.inputs_ready(0), 0));
    CUDA_CHECK(cudaMemcpyAsync(peer.accepted_drafts.data, frame.accepted_drafts.data,
                               frame.accepted_drafts.bytes(), cudaMemcpyDeviceToDevice,
                               rank1.stream));
    ops::speculative_select_accepted_hidden(peer.target_hidden, peer.accepted_drafts,
                                            peer.selected_hidden, rank1.stream);
    ops::scatter(peer.selected_hidden, peer.state_destination_slots,
                 peer_continuation_hidden_store, rank1.stream);
}
} // namespace ninfer::models::qwen3_5::execution
