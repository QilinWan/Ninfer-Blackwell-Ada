#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

/**
 * Vocabulary-split argmax over two tensor-parallel ranks, in three steps: each rank packs its own
 * candidate (argmax_split_pack), one allreduce_sum of the packed candidates gives both ranks both
 * candidates, and a rank selects the complete argmax (argmax_split_select). Rank r holds the
 * contiguous row block [r R, (r + 1) R) of the complete [2R,T] logits.
 *
 * The largest row count one rank may hold: a candidate's row travels as two 8-bit digits.
 */
inline constexpr std::int32_t kArgmaxSplitMaxRowsPerRank = 65536;

// BF16 elements per column of the packed candidates: four digits per rank.
inline constexpr std::int32_t kArgmaxSplitCandidateRows = 8;

/**
 * Rank `rank`'s half of the vocabulary-split argmax. `logits` is its contiguous BF16 block [R,T]
 * with 1 <= R <= kArgmaxSplitMaxRowsPerRank, and `local` the contiguous I32 [T] argmax() of that
 * block over all R rows. Writes all of the contiguous BF16 `candidates` [8,T]: column t is zero
 * except elements [4 rank, 4 rank + 4), which hold the base-256 digits
 * (b >> 8, b & 255, l >> 8, l & 255) of the BF16 bits b of logits[l,t] and of the row l = local[t].
 * Each digit is an integer in [0,255], exactly representable in BF16, and x + 0 is exact, so the
 * allreduce_sum of both ranks' candidates is their exact union on both ranks. `rank` is 0 or 1;
 * `candidates` must not overlap `logits` or `local`. No workspace or other state.
 */
void argmax_split_pack(const Tensor& logits, const Tensor& local, int rank, Tensor& candidates,
                       cudaStream_t stream);

/**
 * The complete argmax from the summed candidates [8,T] of argmax_split_pack over blocks of
 * `rows_per_rank` rows: out[t] = rows_per_rank + l1 when v1 > v0, else l0, where (v_r, l_r) is
 * rank r's candidate. Equal maxima therefore select rank 0's, the lower row, and the result equals
 * argmax() over the complete [2R,T] logits, NaN included, except when row R holds a NaN: argmax()
 * skips a NaN except in its row 0, which row R is in rank 1's block. `out` is contiguous I32 [T]
 * and does not overlap `candidates`. No workspace or other state.
 */
void argmax_split_select(const Tensor& candidates, std::int32_t rows_per_rank, Tensor& out,
                         cudaStream_t stream);

} // namespace ninfer::ops
