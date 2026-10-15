// Translation-unit fragment of src/models/qwen3_5/program/prefill.cpp, included once at its end
// (after its closing namespaces): the two-device (tp 2) vocabulary-split head of a retained
// hidden column. prefill.cpp declares it in its anonymous namespace. Not a header. Our file; see
// docs/maintainer/upstream-merge.md.

#include "models/qwen3_5/execution/tp2/workspace_split.h"

namespace ninfer::models::qwen3_5::execution {
namespace {

// The vocabulary-split output head over one retained hidden column. The hidden axis is
// replicated, so rank 1's operand is rank 0's column itself: rank 1 pulls it once rank 0's stream
// reaches it, each rank then projects its half of the vocabulary, and rank 0 pulls rank 1's half.
void project_split_output_head(PrefillContext& state, const Tensor& hidden, Tensor& logits) {
    const TpExecution& tp             = *state.execution.tp;
    const ExecutionContext& execution = *tp.execution;
    const DeviceContext& rank1        = *execution.dev[1];
    const TextConfig& config          = state.execution.parameters.model.config().text;
    const LinearParameters& head0     = state.execution.parameters.text.output_head;
    const LinearParameters& head1     = tp.parameters->text.output_head;
    const std::int32_t H              = dimension(config.hidden_size);
    tp.work->reset();
    const Tensor rank0_hidden = hidden.view({H, 1});
    const Tensor rank1_hidden = tp.work->alloc(DType::BF16, {H, 1});
    CUDA_CHECK(cudaEventRecord(tp.events->inputs_ready(0), state.execution.device.stream));
    {
        const ScopedCurrentDevice scope(rank1.device);
        CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, tp.events->inputs_ready(0), 0));
        CUDA_CHECK(cudaMemcpyAsync(rank1_hidden.data, rank0_hidden.data, rank1_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, rank1.stream));
    }
    const auto part0 =
        workspace::tp_logits(state.execution.work, head0.weight.n, head1.weight.n, 1, true);
    const auto part1 = workspace::tp_logits(*tp.work, head1.weight.n, head0.weight.n, 1, false);
    output_logits_split_rank0({rank0_hidden, rank1_hidden}, {&head0, &head1},
                              {part0.partial, part1.partial}, logits, part0.staging,
                              {&state.execution.work, tp.work}, execution, *tp.events);
}

} // namespace
} // namespace ninfer::models::qwen3_5::execution
