// Translation-unit fragment of src/models/qwen3_5/program/speculative/target_verification.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) overload of target_verify_accept() (declared in program/context.h).

#include "models/qwen3_5/program/program_impl.h"

namespace ninfer::models::qwen3_5::execution {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          TargetVerifyFrameView peer, Tensor& peer_continuation_hidden_store,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (execution.tp == nullptr) {
        throw std::logic_error("tensor-parallel target verify requires rank 1");
    }
    if (frame.replay_records == nullptr || peer.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    // Proposals, the gathered logits and acceptance are rank 0's alone. The split DFlash2
    // drafter captures rank 1's own features beside rank 0's (peer.feature_sink).
    if (peer.proposal_q.data != nullptr ||
        (peer.feature_sink != nullptr && frame.feature_sink == nullptr)) {
        throw std::logic_error("tensor-parallel target verify accepts on rank 0 only, and captures "
                               "rank 1's features only beside rank 0's");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    card.set_peer_feature_sink(peer.feature_sink);
    const TextContext::RankTensors hidden{frame.target_hidden, peer.target_hidden};
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(
            {frame.ids, peer.ids}, {frame.cache_positions, peer.cache_positions},
            {frame.rope_positions, peer.rope_positions}, {frame.valid_columns, peer.valid_columns},
            {frame.kv_table_rows, peer.kv_table_rows},
            {frame.state_source_slots, peer.state_source_slots}, envelope, hidden,
            frame.target_logits, frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(
            {frame.ids, peer.ids}, {frame.cache_positions, peer.cache_positions},
            {frame.rope_positions, peer.rope_positions}, {frame.valid_columns, peer.valid_columns},
            {frame.kv_table_rows, peer.kv_table_rows},
            {frame.state_source_slots, peer.state_source_slots}, envelope, hidden,
            frame.target_logits, frame.target_tokens);
    }
    card.set_peer_feature_sink(nullptr);
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
    const ScopedCurrentDevice scope(rank1.device);
    CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, events.inputs_ready(0), 0));
    CUDA_CHECK(cudaMemcpyAsync(peer.accepted_drafts.data, frame.accepted_drafts.data,
                               frame.accepted_drafts.bytes(), cudaMemcpyDeviceToDevice,
                               rank1.stream));
    ops::speculative_select_accepted_hidden(peer.target_hidden, peer.accepted_drafts,
                                            peer.selected_hidden, rank1.stream);
    ops::scatter(peer.selected_hidden, peer.state_destination_slots, peer_continuation_hidden_store,
                 rank1.stream);
}

} // namespace ninfer::models::qwen3_5::execution
