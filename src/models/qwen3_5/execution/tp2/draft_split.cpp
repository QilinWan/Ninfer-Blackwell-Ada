#include "models/qwen3_5/execution/tp2/draft_split.h"

#include "core/device_scope.h"
#include "models/qwen3_5/execution/tp2/workspace_split.h"
#include "ninfer/ops/tp2/linear_topk.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

bool dflash2_split_candidates(const Parameters& rank0, const Parameters* rank1) {
    return rank1 != nullptr && rank0.proposal && rank0.proposal->split() && rank1->proposal &&
           rank1->proposal->split();
}

std::size_t dflash2_candidates_split_workspace_bytes(const ProposalParameters& head,
                                                     std::int32_t hidden, std::int32_t columns,
                                                     bool broadcast) {
    WorkspaceLayoutBuilder layout;
    (void)workspace::tp_dflash2_candidates(layout, hidden, columns, broadcast);
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_topk_split_workspace_capacity_bytes(
            head.head.weight.qtype, head.head.weight.n, head.head.weight.k, columns, columns));
    }
    return layout.peak_bytes(1);
}

void dflash2_candidates_split(const std::array<Tensor, 2>& hidden,
                              const std::array<const ProposalParameters*, 2>& head, bool broadcast,
                              Tensor& ids, Tensor& scores,
                              const std::array<WorkspaceArena*, 2>& workspace,
                              const ExecutionContext& execution, const ops::PeerEvents& events,
                              Tensor* peer_ids, Tensor* peer_scores) {
    const std::int32_t rows    = hidden[0].ne[0];
    const std::int32_t columns = hidden[0].ne[1];
    for (std::size_t r = 0; r < 2; ++r) {
        if (head[r] == nullptr || !head[r]->split() || !head[r]->token_ids ||
            head[r]->token_ids->ne[0] != head[r]->head.weight.n) {
            throw std::logic_error("tensor-parallel DFlash2 candidates need both head blocks");
        }
        if (hidden[r].dtype != DType::BF16 || hidden[r].ne[0] != rows ||
            hidden[r].ne[1] != columns || !hidden[r].is_contiguous()) {
            throw std::invalid_argument("tensor-parallel DFlash2 candidates: hidden differ");
        }
    }
    if (!events.live()) {
        throw std::invalid_argument("tensor-parallel DFlash2 candidates: dead events");
    }
    auto scope0 = workspace[0]->scope();
    auto scope1 = workspace[1]->scope();
    const std::array<workspace::TensorParallelCandidateRoots, 2> roots{
        workspace::tp_dflash2_candidates(*workspace[0], rows, columns, broadcast),
        workspace::tp_dflash2_candidates(*workspace[1], rows, columns, broadcast)};
    std::array<Tensor, 2> local_hidden = hidden;
    if (broadcast) {
        local_hidden[1] = roots[1].hidden;
        {
            const ScopedCurrentDevice rank1(execution.dev[1]->device);
            CUDA_CHECK(cudaMemsetAsync(local_hidden[1].data, 0, local_hidden[1].bytes(),
                                       execution.dev[1]->stream));
        }
        ops::allreduce_sum(local_hidden, {roots[0].hidden_staging, roots[1].hidden_staging},
                           execution, events);
    }
    std::array<Tensor, 2> packed{roots[0].candidates, roots[1].candidates};
    {
        const ScopedCurrentDevice restore;
        for (int rank = 0; rank < 2; ++rank) {
            const auto r               = static_cast<std::size_t>(rank);
            const DeviceContext& owner = *execution.dev[r];
            ScopedCurrentDevice::select(owner.device);
            Tensor local_ids    = roots[r].ids;
            Tensor local_scores = roots[r].scores;
            ops::linear_topk_split(local_hidden[r], head[r]->head.weight, *head[r]->token_ids,
                                   local_ids, local_scores, *workspace[r], owner.stream);
            ops::topk_split_pack(local_ids, local_scores, rank, packed[r], owner.stream);
        }
    }
    // The ranks' candidates occupy disjoint digits, so the summing exchange is their exact union.
    ops::allreduce_sum(packed, {roots[0].staging, roots[1].staging}, execution, events);
    if ((peer_ids == nullptr) != (peer_scores == nullptr)) {
        throw std::invalid_argument(
            "tensor-parallel DFlash2 candidates: rank 1 needs both ids and scores");
    }
    const ScopedCurrentDevice restore;
    ScopedCurrentDevice::select(execution.dev[0]->device);
    ops::topk_split_merge(packed[0], ids, scores, execution.dev[0]->stream);
    if (peer_ids != nullptr) {
        // Rank 1 holds the same union: its merge writes the same top sixteen.
        ScopedCurrentDevice::select(execution.dev[1]->device);
        ops::topk_split_merge(packed[1], *peer_ids, *peer_scores, execution.dev[1]->stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
