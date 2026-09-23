#include "models/qwen3_5/execution/ffn.h"

#include "core/layout.h"
#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    if (mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp) {
    const auto stream  = execution.stream;
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        execution);
        return;
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, p.gate_up.policy, workspace, stream);
    }
    ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
}

namespace {

const DenseParameters& split_dense(const FfnParameters& parameters) {
    const auto* dense = std::get_if<DenseParameters>(&parameters);
    if (dense == nullptr) {
        throw std::invalid_argument("tensor-parallel FFN: the MoE FFN has no two-device route");
    }
    return *dense;
}

} // namespace

std::size_t ffn_split_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                      std::int32_t last) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    const auto& p    = split_dense(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_row_parallel_workspace_capacity_bytes(
            down.qtype, down.n, down.k, p.down.policy, first, last));
    }
    return layout.peak_bytes(1);
}

void ffn_split(const std::array<Tensor, 2>& hidden,
               const std::array<const FfnParameters*, 2>& parameters,
               const std::array<Tensor, 2>& residual, const std::array<Tensor, 2>& staging,
               const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution,
               const ops::PeerEvents& events) {
    const DenseParameters& p0 = split_dense(*parameters[0]);
    const DenseParameters& p1 = split_dense(*parameters[1]);
    auto scope0               = workspace[0]->scope();
    auto scope1               = workspace[1]->scope();
    const auto columns        = hidden[0].ne[1];
    const std::array<Tensor, 2> activation{
        workspace[0]->alloc(DType::BF16, {p0.gate_up.weight.n / 2, columns}),
        workspace[1]->alloc(DType::BF16, {p1.gate_up.weight.n / 2, columns})};
    project_swiglu_column_parallel(hidden, {&p0.gate_up, &p1.gate_up}, activation, workspace,
                                   execution);
    project_add_row_parallel(activation, {&p0.down, &p1.down}, residual, staging, workspace,
                             execution, events);
}

std::size_t mtp_ffn_split_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                          std::int32_t last) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    const auto& p    = split_dense(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
    }
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    (void)layout.alloc(DType::BF16, {down.n, last});
    (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                  p.down.policy, first, last));
    return layout.peak_bytes(1);
}

void mtp_ffn_split(const std::array<Tensor, 2>& hidden,
                   const std::array<const FfnParameters*, 2>& parameters,
                   const std::array<Tensor, 2>& residual, const std::array<Tensor, 2>& staging,
                   const std::array<WorkspaceArena*, 2>& workspace,
                   const ExecutionContext& execution, const ops::PeerEvents& events) {
    const std::array<const DenseParameters*, 2> p{&split_dense(*parameters[0]),
                                                  &split_dense(*parameters[1])};
    auto scope0        = workspace[0]->scope();
    auto scope1        = workspace[1]->scope();
    const auto columns = hidden[0].ne[1];
    std::array<Tensor, 2> gate_up;
    std::array<Tensor, 2> activation;
    std::array<Tensor, 2> delta;
    for (std::size_t r = 0; r < 2; ++r) {
        gate_up[r] = workspace[r]->alloc(DType::BF16, {p[r]->gate_up.weight.n, columns});
    }
    project_column_parallel(hidden, {&p[0]->gate_up, &p[1]->gate_up}, gate_up, workspace,
                            execution);
    for (std::size_t r = 0; r < 2; ++r) {
        activation[r] = workspace[r]->alloc(DType::BF16, {p[r]->gate_up.weight.n / 2, columns});
        delta[r]      = workspace[r]->alloc(DType::BF16, {p[r]->down.weight.n, columns});
    }
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    for (std::size_t r = 0; r < 2; ++r) {
        const std::int32_t half = p[r]->gate_up.weight.n / 2;
        CUDA_CHECK(cudaSetDevice(execution.dev[r]->device));
        ops::silu_mul(gate_up[r].slice(0, 0, half), gate_up[r].slice(0, half, half), activation[r],
                      execution.dev[r]->stream);
    }
    CUDA_CHECK(cudaSetDevice(previous));
    project_row_parallel(activation, {&p[0]->down, &p[1]->down}, delta, staging, workspace,
                         execution, events);
    for (std::size_t r = 0; r < 2; ++r) {
        CUDA_CHECK(cudaSetDevice(execution.dev[r]->device));
        Tensor target = residual[r];   // Tensor is a view: residual_add wants a mutable lvalue
        ops::residual_add(delta[r], target, execution.dev[r]->stream);
    }
    CUDA_CHECK(cudaSetDevice(previous));
}

} // namespace ninfer::models::qwen3_5::execution
