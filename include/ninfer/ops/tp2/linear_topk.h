#pragma once

// Two-device (tp 2) forms of linear_topk. Our file: ninfer/ops/linear_topk.h is upstream's and
// carries no hook.

#include "ninfer/ops/linear_topk.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Vocabulary-split top sixteen of the optimized proposal head over two tensor-parallel ranks, in
 * three steps: each rank ranks its own row block (linear_topk_split) and packs its sixteen
 * candidates (topk_split_pack); one allreduce_sum of the packed candidates gives both ranks both
 * lists; a rank merges them (topk_split_merge). The Q4_G64_FP16 `[131072,5120]` head splits into
 * two `[65536,5120]` row blocks, rank r holding rows [65536 r, 65536 (r + 1)) and the matching
 * block of the row -> global token id map.
 *
 * Every candidate is the total-order key of linear_topk (FP32 score bits, then the lower global
 * id on equal scores; ops/common/score_id_order.cuh). A block row computes the whole head's row
 * bit for bit (the row tiles of the Q4 kernels reduce each row alone over the same K slices), and
 * the keys are distinct because the global ids are, so the top sixteen keys of the union of the two
 * blocks' top sixteen are the whole head's top sixteen, ties included.
 */

// Rows of one rank's block of the optimized proposal head.
inline constexpr std::int32_t kLinearTopKSplitRows = 65536;

// BF16 elements per column of the packed candidates: 16 keys of 8 base-256 digits per rank.
inline constexpr std::int32_t kTopKSplitCandidateRows = 2 * 16 * 8;

/**
 * The caller-owned transient capacity linear_topk_split needs for every column count in the
 * inclusive `[min_columns,max_columns]` interval. Only the Q4_G64_FP16 `[65536,5120]` block is
 * registered.
 */
[[nodiscard]] std::size_t linear_topk_split_workspace_capacity_bytes(QType qtype,
                                                                     std::int32_t head_rows,
                                                                     std::int32_t input_rows,
                                                                     std::int32_t min_columns,
                                                                     std::int32_t max_columns);

/**
 * One rank's block of the vocabulary-split top sixteen: linear_topk's optimized-head contract with
 * `head` Q4_G64_FP16 `[65536,5120]` (a row block of the `[131072,5120]` head) and
 * `row_to_global_ids` I32 `[65536]` (the same block of the head's map). The output is the top
 * sixteen of the block's rows in linear_topk's order and encoding.
 */
void linear_topk_split(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                       Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                       cudaStream_t stream);

/**
 * Packs rank `rank`'s sixteen candidates, `candidate_ids` I32 `[16,U]` and `candidate_scores`
 * FP32 `[16,U]` as linear_topk_split writes them, into the contiguous BF16 `candidates`
 * `[kTopKSplitCandidateRows,U]`. Column u is zero except its elements [128 rank, 128 rank + 128):
 * candidate i's 64-bit key as eight base-256 digits, most significant first, at 128 rank + 8 i.
 * Each digit is an integer in [0,255], exact in BF16, and x + 0 is exact, so allreduce_sum of both
 * ranks' packed candidates is their exact union on both ranks. `rank` is 0 or 1; no overlap.
 */
void topk_split_pack(const Tensor& candidate_ids, const Tensor& candidate_scores, int rank,
                     Tensor& candidates, cudaStream_t stream);

/**
 * The complete top sixteen from the summed `candidates` `[kTopKSplitCandidateRows,U]` of
 * topk_split_pack: the sixteen largest of the 32 keys of each column, in descending key order,
 * decoded into `candidate_ids` I32 `[16,U]` and `candidate_scores` FP32 `[16,U]` exactly as
 * linear_topk decodes its keys. No workspace or other state; outputs must not overlap the input.
 */
void topk_split_merge(const Tensor& candidates, Tensor& candidate_ids, Tensor& candidate_scores,
                      cudaStream_t stream);

} // namespace ninfer::ops
