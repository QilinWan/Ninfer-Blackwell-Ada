#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the caller-owned transient capacity required by linear_topk for every column count in the
 * inclusive `[min_columns,max_columns]` interval. The registered profile is identified by
 * its weight format and exact `[head_rows,input_rows]` geometry. Column counts are positive.
 */
[[nodiscard]] std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                               std::int32_t input_rows,
                                                               std::int32_t min_columns,
                                                               std::int32_t max_columns);

/**
 * @brief Projects independent matrix columns through a full vocabulary head and returns
 * the stable top sixteen scores and global token ids per column.
 *
 * @details For any positive column count `U`, `hidden` is contiguous BF16 `[5120,U]`, `head` is
 * either Q8_G32_FP16 or FP8_E4M3FN_ROW_BF16 `[248320,5120]`, and `valid_rows` is 248077. For every
 * `t in [0,U)` and valid vocabulary row `v`, the ideal score is
 *
 * @f[
 *   s_{v,t}=\sum_{k=0}^{5119}\mathrm{FP32Dequant}(head)_{v,k}
 *                              \mathrm{FP32}(hidden_{k,t}).
 * @f]
 *
 * `candidate_scores` is contiguous FP32 `[16,U]` and `candidate_ids` is contiguous I32
 * `[16,U]`. Rank is stored fastest; a caller may view the output as `[16,K,B]` when `U=K*B`.
 * Candidates are ordered by descending computed score, with exact score ties resolved by lower
 * global token id. Physical rows `[valid_rows,248320)` never participate. Projection uses the
 * registered A16 arithmetic profile; candidate scores are returned directly in FP32 and no dense
 * vocabulary-logit tensor is an observable intermediate.
 *
 * All tensors and every weight plane are preserved except that both output tensors are completely
 * overwritten. Inputs, outputs, weight planes, and workspace must not overlap. The Op has no
 * persistent state and performs no internal device allocation.
 */
void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream);

/**
 * @brief Projects through the optimized 131072-row proposal head and returns stable top sixteen
 * scores with shortlist rows mapped to global token ids.
 *
 * @details The tensor contract is the same as the full-head overload except that `head` is
 * Q4_G64_FP16 `[131072,5120]` and every head row participates. `row_to_global_ids` is contiguous
 * I32 `[131072]`; it contains distinct ids in `[0,248077)` and maps each local head row to the id
 * used for output and tie-breaking. Artifact binding establishes the map's range and uniqueness.
 */
void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream);

/**
 * @brief Vocabulary-split top sixteen of the optimized proposal head over two tensor-parallel
 * ranks (tp 2, the split DFlash2 drafter: load/sharding.h NINFER_TP_DRAFTER=split).
 *
 * @details Three steps: each rank ranks its own row block (linear_topk_split) and packs its
 * sixteen candidates (topk_split_pack); one allreduce_sum of the packed candidates gives both
 * ranks both lists; a rank merges them (topk_split_merge). The Q4_G64_FP16 `[131072,5120]` head
 * splits into two `[65536,5120]` row blocks, rank r holding rows `[65536 r, 65536 (r + 1))` and
 * the matching block of the row -> global token id map.
 *
 * Every candidate is the total-order key of linear_topk (FP32 score bits, then the lower global
 * id on equal scores; ops/common/score_id_order.cuh). A block row computes the whole head's row
 * bit for bit (the row tiles of the Q4 kernels reduce each row alone over the same K slices), and
 * the keys are distinct because the global ids are, so the top sixteen keys of the union of the
 * two blocks' top sixteen are the whole head's top sixteen, ties included.
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
 * One rank's block of the vocabulary-split top sixteen: linear_topk's optimized-head contract
 * with `head` Q4_G64_FP16 `[65536,5120]` (a row block of the `[131072,5120]` head) and
 * `row_to_global_ids` I32 `[65536]` (the same block of the head's map). The output is the top
 * sixteen of the block's rows in linear_topk's order and encoding.
 */
void linear_topk_split(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                       Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                       cudaStream_t stream);

/**
 * Packs rank `rank`'s sixteen candidates, `candidate_ids` I32 `[16,U]` and `candidate_scores`
 * FP32 `[16,U]` as linear_topk_split writes them, into the contiguous BF16 `candidates`
 * `[kTopKSplitCandidateRows,U]`. Column u is zero except its elements `[128 rank, 128 rank +
 * 128)`: candidate i's 64-bit key as eight base-256 digits, most significant first, at
 * `128 rank + 8 i`. Each digit is an integer in [0,255], exact in BF16, and x + 0 is exact, so
 * allreduce_sum of both ranks' packed candidates is their exact union on both ranks. `rank` is
 * 0 or 1; no overlap.
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
