// Translation-unit fragment of src/ops/linear_topk/linear_topk.cpp, included once at its end
// (after its closing namespace), so the two-device code below shares that file's private helpers
// (validation, plans, the chunked launch) without copying them. Not a header. Our file; see
// docs/maintainer/upstream-merge.md.
//
// The vocabulary-split block of the optimized proposal head, declared in
// ninfer/ops/tp2/linear_topk.h. A block is the parent's problem with half the rows and the same K:
// it takes the parent's plan (plan_for(Q4Optimized, columns)) and the parent's launchers, whose row
// tiles reduce each row alone, so every block row computes the whole head's row bit for bit.

#include "ninfer/ops/tp2/linear_topk.h"

namespace ninfer::ops {
namespace {

void require_q4_block(const Weight& head) {
    const bool common =
        head.qtype == QType::Q4_G64_FP16 && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 64 && head.group == 64 &&
        head.ndim == 2 && head.n == kLinearTopKSplitRows && head.k == detail::kLinearTopKHidden &&
        head.shape[0] == head.n && head.shape[1] == head.k && head.padded_shape[0] == head.n &&
        head.padded_shape[1] == head.k && head.qhigh == nullptr && head.high_plane_bytes == 0 &&
        aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk_split: invalid Q4 head block"); }
}

void require_split_profile(QType qtype, std::int32_t head_rows, std::int32_t input_rows) {
    if (qtype != QType::Q4_G64_FP16 || head_rows != kLinearTopKSplitRows ||
        input_rows != detail::kLinearTopKHidden) {
        throw std::invalid_argument("linear_topk_split: unsupported head block profile");
    }
}

void execute_block(const Tensor& hidden, const Weight& head, const Tensor& id_map, Tensor& ids,
                   Tensor& scores, WorkspaceArena& workspace, cudaStream_t stream) {
    for (int first = 0; first < hidden.ne[1];) {
        const int columns  = std::min(detail::kLinearTopKMaxChunkColumns, hidden.ne[1] - first);
        auto x             = column_slice(hidden, first, columns);
        auto out_ids       = column_slice(ids, first, columns);
        auto out_scores    = column_slice(scores, first, columns);
        const auto plan    = plan_for(HeadProfile::Q4Optimized, columns);
        auto scope         = workspace.scope();
        const auto scratch = detail::allocate_linear_topk_workspace(
            workspace, head.n, columns, plan.rows, plan.tile, plan.block_k);
        require_scratch_nonoverlap(hidden, ids, scores, &id_map, scratch);
        require_no_weight_overlap(head, hidden, ids, scores, &id_map, scratch);
        detail::linear_topk_q4_launch(x, head, id_map, scratch, stream);
        detail::linear_topk_merge_launch(scratch, out_ids, out_scores, stream);
        first += columns;
    }
}

} // namespace

std::size_t linear_topk_split_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                       std::int32_t input_rows,
                                                       std::int32_t min_columns,
                                                       std::int32_t max_columns) {
    require_split_profile(qtype, head_rows, input_rows);
    if (min_columns < 1 || max_columns < min_columns)
        throw std::invalid_argument("linear_topk_split workspace: invalid column interval");
    WorkspaceLayoutBuilder layout;
    for (int columns = 1; columns <= detail::kLinearTopKMaxChunkColumns; ++columns) {
        const bool reachable = (columns >= min_columns && columns <= max_columns) ||
                               (max_columns > detail::kLinearTopKMaxChunkColumns);
        if (!reachable) continue;
        const auto plan = plan_for(HeadProfile::Q4Optimized, columns);
        auto scope      = layout.scope();
        (void)detail::allocate_linear_topk_workspace(layout, head_rows, columns, plan.rows,
                                                     plan.tile, plan.block_k);
    }
    return layout.peak_bytes();
}

void linear_topk_split(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                       Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    validate_io(hidden, candidate_ids, candidate_scores);
    require_split_profile(head.qtype, head.n, head.k);
    require_q4_block(head);
    require_matrix(row_to_global_ids, DType::I32, kLinearTopKSplitRows, 1, "row_to_global_ids", 4);
    if (overlaps(hidden, row_to_global_ids) || overlaps(candidate_ids, row_to_global_ids) ||
        overlaps(candidate_scores, row_to_global_ids)) {
        throw std::invalid_argument("linear_topk_split: id map overlaps input or output");
    }
    execute_block(hidden, head, row_to_global_ids, candidate_ids, candidate_scores, workspace,
                  stream);
}

} // namespace ninfer::ops
