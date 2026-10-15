// Translation-unit fragment of src/ops/wrapper/argmax.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// Two-device (tp 2) forms of argmax, declared in ninfer/ops/tp2/argmax.h.

#include "ninfer/ops/tp2/argmax.h"
#include "ops/launcher/argmax_split.h"

namespace ninfer::ops {

namespace {

void require_split_vector(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t columns,
                          const char* op, const char* label) {
    if (t.dtype != dtype || t.ne[0] != rows || t.ne[1] != columns || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + label + " must be contiguous " +
                                    (dtype == DType::BF16 ? "BF16 [" : "I32 [") +
                                    std::to_string(rows) + "," + std::to_string(columns) + "]");
    }
}

} // namespace

void argmax_split_pack(const Tensor& logits, const Tensor& local, int rank, Tensor& candidates,
                       cudaStream_t stream) {
    constexpr const char* op = "argmax_split_pack";
    const std::int32_t rows  = logits.ne[0];
    const std::int32_t cols  = logits.ne[1];
    if (rows <= 0 || rows > kArgmaxSplitMaxRowsPerRank || cols <= 0) {
        throw std::invalid_argument("argmax_split_pack: logits must be [R,T] with 1 <= R <= 65536 "
                                    "and T >= 1");
    }
    if (rank != 0 && rank != 1) {
        throw std::invalid_argument("argmax_split_pack: rank must be 0 or 1");
    }
    require_split_vector(logits, DType::BF16, rows, cols, op, "logits");
    require_split_vector(local, DType::I32, cols, 1, op, "local");
    require_split_vector(candidates, DType::BF16, kArgmaxSplitCandidateRows, cols, op,
                         "candidates");
    detail::argmax_split_pack_launch(logits, local, rank, candidates, stream);
}

void argmax_split_select(const Tensor& candidates, std::int32_t rows_per_rank, Tensor& out,
                         cudaStream_t stream) {
    constexpr const char* op = "argmax_split_select";
    const std::int32_t cols  = candidates.ne[1];
    if (rows_per_rank <= 0 || rows_per_rank > kArgmaxSplitMaxRowsPerRank || cols <= 0) {
        throw std::invalid_argument(
            "argmax_split_select: rows_per_rank must be in [1,65536] and T >= 1");
    }
    require_split_vector(candidates, DType::BF16, kArgmaxSplitCandidateRows, cols, op,
                         "candidates");
    require_split_vector(out, DType::I32, cols, 1, op, "out");
    detail::argmax_split_select_launch(candidates, rows_per_rank, out, stream);
}

} // namespace ninfer::ops
