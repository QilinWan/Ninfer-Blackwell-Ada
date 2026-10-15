// Implements: include/ninfer/ops/tp2/linear_topk.h (topk_split_pack, topk_split_merge)
// Match: contiguous I32/FP32 [16,U] candidates of linear_topk_split, BF16 [256,U] packed
// candidates. Our file.
// Algorithm assumptions: a candidate is linear_topk's 64-bit total-order key, recomputed from its
// decoded (score, id) pair (score_id_order_key inverts score_from_order_key/id_from_order_key,
// including the INT_MAX sentinel); eight base-256 digits per key are integers BF16 represents
// exactly, so the summing exchange of the two ranks' disjoint digits is exact. The merge sorts the
// 32 keys of a column with one warp's bitonic network under linear_topk's order.
#include "ninfer/ops/tp2/linear_topk.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/common/score_id_order.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kTopK          = 16;
constexpr int kDigits        = 8;
constexpr int kRankRows      = kTopK * kDigits; // 128 BF16 per rank and column
constexpr int kMergeColumns  = 4;               // one warp per column

static_assert(kTopKSplitCandidateRows == 2 * kRankRows);

__global__ void topk_split_pack_kernel(const std::int32_t* __restrict__ ids,
                                       const float* __restrict__ scores, int rank,
                                       std::int32_t columns, __nv_bfloat16* __restrict__ out) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const int j               = static_cast<int>(threadIdx.x); // [0, 256)
    if (column >= columns) { return; }
    float value = 0.0F;
    const int local = j - rank * kRankRows;
    if (local >= 0 && local < kRankRows) {
        const int candidate = local / kDigits;
        const int digit     = local % kDigits;
        const std::int64_t at = static_cast<std::int64_t>(column) * kTopK + candidate;
        const std::uint64_t key = score_id_order_key(scores[at], ids[at]);
        value = static_cast<float>((key >> (8 * (kDigits - 1 - digit))) & 0xffU);
    }
    out[static_cast<std::int64_t>(column) * kTopKSplitCandidateRows + j] =
        __float2bfloat16_rn(value);
}

__global__ void topk_split_merge_kernel(const __nv_bfloat16* __restrict__ candidates,
                                        std::int32_t columns, std::int32_t* __restrict__ ids,
                                        float* __restrict__ scores) {
    const int lane            = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x) * kMergeColumns +
                                static_cast<std::int32_t>(threadIdx.x >> 5);
    if (column >= columns) { return; }
    // Lane l holds candidate l % 16 of rank l / 16.
    const __nv_bfloat16* c =
        candidates + static_cast<std::int64_t>(column) * kTopKSplitCandidateRows + lane * kDigits;
    std::uint64_t key = 0;
#pragma unroll
    for (int d = 0; d < kDigits; ++d) {
        key = (key << 8) | (static_cast<std::uint64_t>(__bfloat162float(c[d])) & 0xffU);
    }
    // Bitonic sort, descending across the warp: lane 0 ends with the largest key.
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const std::uint64_t other = __shfl_xor_sync(0xffffffffU, key, j);
            const bool lower          = (lane & j) == 0;
            const bool descending     = (lane & k) == 0;
            const std::uint64_t hi    = key > other ? key : other;
            const std::uint64_t lo    = key > other ? other : key;
            key                       = lower == descending ? hi : lo;
        }
    }
    if (lane < kTopK) {
        const std::int64_t at = static_cast<std::int64_t>(column) * kTopK + lane;
        ids[at]               = id_from_order_key(key);
        scores[at]            = score_from_order_key(key);
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* op, const char* label) {
    if (t.dtype != dtype || t.ne[0] != rows || t.ne[1] != columns || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + label + " must be contiguous [" +
                                    std::to_string(rows) + "," + std::to_string(columns) + "]");
    }
}

} // namespace

void topk_split_pack(const Tensor& candidate_ids, const Tensor& candidate_scores, int rank,
                     Tensor& candidates, cudaStream_t stream) {
    constexpr const char* op  = "topk_split_pack";
    const std::int32_t columns = candidate_ids.ne[1];
    if (columns <= 0) { throw std::invalid_argument("topk_split_pack: no columns"); }
    if (rank != 0 && rank != 1) { throw std::invalid_argument("topk_split_pack: rank must be 0 or 1"); }
    require_matrix(candidate_ids, DType::I32, kTopK, columns, op, "candidate_ids");
    require_matrix(candidate_scores, DType::FP32, kTopK, columns, op, "candidate_scores");
    require_matrix(candidates, DType::BF16, kTopKSplitCandidateRows, columns, op, "candidates");
    if (overlaps(candidates, candidate_ids) || overlaps(candidates, candidate_scores)) {
        throw std::invalid_argument("topk_split_pack: candidates overlap an input");
    }
    topk_split_pack_kernel<<<static_cast<unsigned>(columns), kTopKSplitCandidateRows, 0, stream>>>(
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const float*>(candidate_scores.data), rank, columns,
        static_cast<__nv_bfloat16*>(candidates.data));
    CUDA_CHECK(cudaGetLastError());
}

void topk_split_merge(const Tensor& candidates, Tensor& candidate_ids, Tensor& candidate_scores,
                      cudaStream_t stream) {
    constexpr const char* op  = "topk_split_merge";
    const std::int32_t columns = candidates.ne[1];
    if (columns <= 0) { throw std::invalid_argument("topk_split_merge: no columns"); }
    require_matrix(candidates, DType::BF16, kTopKSplitCandidateRows, columns, op, "candidates");
    require_matrix(candidate_ids, DType::I32, kTopK, columns, op, "candidate_ids");
    require_matrix(candidate_scores, DType::FP32, kTopK, columns, op, "candidate_scores");
    if (overlaps(candidates, candidate_ids) || overlaps(candidates, candidate_scores) ||
        overlaps(candidate_ids, candidate_scores)) {
        throw std::invalid_argument("topk_split_merge: operands overlap");
    }
    topk_split_merge_kernel<<<static_cast<unsigned>(div_up(columns, kMergeColumns)),
                              32 * kMergeColumns, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(candidates.data), columns,
        static_cast<std::int32_t*>(candidate_ids.data), static_cast<float*>(candidate_scores.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
