#pragma once

// Two-device (tp 2) workspace recipes not carried by execution/workspace.h: the vocabulary-split
// DFlash2 candidate ranking roots. The tp call, logits, proposal argmax and MTP stem recipes are
// already in the upstream header's tensor-parallel section.

#include "models/qwen3_5/execution/workspace.h"
#include "ninfer/ops/linear_topk.h"

#include <cstdint>

namespace ninfer::models::qwen3_5::execution::workspace {

// The vocabulary-split DFlash2 candidate ranking over `columns` final drafter hidden columns on one
// rank (execution::dflash2_candidates_split): with `broadcast`, the hidden rank 1 receives from
// rank 0 and the all-reduce staging it arrives through (rank 1's `hidden` is its copy, rank 0's is
// unused); then the rank's top sixteen, its packed candidates and their all-reduce staging.
struct TensorParallelCandidateRoots {
    Tensor hidden;
    Tensor hidden_staging;
    Tensor ids;
    Tensor scores;
    Tensor candidates;
    Tensor staging;
};

template <class Allocator>
TensorParallelCandidateRoots tp_dflash2_candidates(Allocator& allocator, std::int32_t hidden,
                                                   std::int32_t columns, bool broadcast) {
    constexpr std::int32_t top_k = 16;
    TensorParallelCandidateRoots out;
    if (broadcast) {
        out.hidden         = matrix(allocator, DType::BF16, hidden, columns);
        out.hidden_staging = matrix(allocator, DType::BF16, hidden, columns);
    }
    out.ids        = matrix(allocator, DType::I32, top_k, columns);
    out.scores     = matrix(allocator, DType::FP32, top_k, columns);
    out.candidates = matrix(allocator, DType::BF16, ops::kTopKSplitCandidateRows, columns);
    out.staging    = matrix(allocator, DType::BF16, ops::kTopKSplitCandidateRows, columns);
    return out;
}

} // namespace ninfer::models::qwen3_5::execution::workspace
