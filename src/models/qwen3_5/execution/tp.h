#pragma once

// Two-device tensor-parallel (tp == 2) execution operands of the Text backbone.
//
// Rank r of a tp2 Model holds the head-, intermediate- and vocabulary-split shards described in
// load/sharding.h and executes on ExecutionContext::dev[r]. The hidden/residual axis is
// replicated: every layer closes with a row-parallel projection whose all-reduce leaves the same
// BF16 residual on both ranks, so norms, the token embedding and every per-rank state update see
// identical inputs. Rank 0 alone owns request bookkeeping, sampling and the round egress; rank 1
// owns only its shards of the weights, KV planes, GDN state and scratch.

#include "core/arena.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/allreduce.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

inline constexpr int kTensorParallelWidth = 2;

// The logical Text config narrowed to one rank's share: attention query/KV heads, GDN key/value
// heads, the Dense intermediate width and the vocabulary are divided by `width`; the hidden size,
// head dimensions, layer schedule and RoPE are unchanged. It names the per-rank extents of the
// shared workspace recipes, KV planes and GDN state pools. Throws std::invalid_argument for an
// extent `width` does not divide and for the MoE FFN, which has no tensor-parallel placement.
[[nodiscard]] TextConfig shard_text_config(const TextConfig& config, int width);

// Rank 1's copy of the ordinary decode control. The Program publishes it on rank 1's stream before
// the call, by uploading the same OrdinaryDecodeIngress record rank 0 receives into a frame of
// the same layout on rank 1; only the first B entries of each vector are read. The fields name
// the same tokens, positions and state slots as rank 0's, and the same KV execution rows: rank
// 0's KVExecutionTablePool replays every acquire and publication onto rank 1's mirror at the same
// row index. Rank 1 never samples, so the ingress sampling configs are never read there.
struct OrdinaryPeerFrame {
    Tensor tokens;                  // I32 [capacity]
    Tensor cache_positions;         // I32 [capacity]
    Tensor rope_positions;          // I32 [capacity]
    Tensor text_kv_table_rows;      // I32 [capacity]
    Tensor state_source_slots;      // I32 [capacity]
    Tensor state_destination_slots; // I32 [capacity]
};

// Views a rank-1 OrdinaryDecodeState's ingress fields; the frame must be resident on rank 1.
[[nodiscard]] OrdinaryPeerFrame ordinary_peer_frame(const qwen3_5::OrdinaryDecodeState& frame);

// Vocabulary-split output head. Rank r projects `hidden[r]` [H,C] through its head shard
// [V_r,H] into `partial[r]` [V_r,C]; one exact row gather per column then assembles the complete
// [V,C] logits in both `logits[0]` (rank 0, the one consumer) and `logits[1]` (rank 1 scratch,
// written because the gather leaves its image on both ranks). V = V_0 + V_1, rank 0 first.
[[nodiscard]] std::size_t output_head_split_workspace_bytes(const LinearParameters& shard,
                                                            std::int32_t first, std::int32_t last);
void output_logits_split(const std::array<Tensor, 2>& hidden,
                         const std::array<const LinearParameters*, 2>& head,
                         const std::array<Tensor, 2>& partial, const std::array<Tensor, 2>& logits,
                         const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& execution, const ops::PeerEvents& events);

// Everything a TextContext needs to drive rank 1 in lockstep with its own rank-0 operands. The
// TextContext's DeviceContext must be `execution->dev[0]`. All members are borrowed and must
// outlive the context; the pointers into Program storage are stable for the Program's lifetime.
struct TpExecution {
    const ExecutionContext* execution = nullptr; // tp == 2
    const ops::PeerEvents* events     = nullptr; // one instance per stream pair, Program-owned
    const Parameters* parameters      = nullptr; // Parameters(model, 1)
    WorkspaceArena* work              = nullptr; // rank 1's transient arena
    // Rank 1's GDN state pool (value heads / 2, conv channels / 2), slot-for-slot with rank 0's.
    LinearAttentionStatePool* linear_attention = nullptr;
    // Rank 1's text KV cache (KV heads / 2); its page pool and execution tables are rank 0's
    // mirrors, so a rank-0 execution row names the same pages on rank 1.
    const qwen3_5::PagedKVCache* text_cache = nullptr;
    // I32 [1] on rank 1: the prefill execution row, equal to rank 0's RoundState
    // text_kv_table_row. Read by prefill only.
    Tensor text_kv_table_row;
    // Rank 1's ordinary decode control. Read by ordinary decode only; may be null otherwise.
    const OrdinaryPeerFrame* ordinary = nullptr;

    [[nodiscard]] bool complete() const noexcept {
        return execution != nullptr && events != nullptr && parameters != nullptr &&
               work != nullptr && linear_attention != nullptr && text_cache != nullptr;
    }
};

} // namespace ninfer::models::qwen3_5::execution
