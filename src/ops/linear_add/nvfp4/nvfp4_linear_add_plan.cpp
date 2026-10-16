#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    A4,
};

// The two-device input-column halves [5120,3072] and [5120,8704] keep the crossover of the problem
// they halve, and linear() over each half uses the same one (shapes/n5120_k3072.cu and
// n5120_k8704.cu), so both ranks of a row-parallel projection take the same route.
Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    const bool output_family =
        input_rows == Nvfp4N5120K6144::kInputRows || input_rows == Nvfp4N5120K3072::kInputRows;
    const bool down_family =
        input_rows == Nvfp4N5120K17408::kInputRows || input_rows == Nvfp4N5120K8704::kInputRows;
    if (tokens <= 0 || output_rows != Nvfp4N5120K6144::kOutputRows ||
        (!output_family && !down_family)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4LinearAddRoute::A16;
    }
    if (!allows_a4(policy)) { throw std::invalid_argument("nvfp4 linear_add: unsupported policy"); }
    if (input_rows == Nvfp4N5120K3072::kInputRows) {
        return Nvfp4LinearAddRoute::A16;  // no A4 route for this half in this tree
    }
    const std::int32_t first_a4 =
        output_family ? kNvfp4OutputFamilyFirstA4Tokens : kNvfp4DownFamilyFirstA4Tokens;
    return tokens >= first_a4 ? Nvfp4LinearAddRoute::A4 : Nvfp4LinearAddRoute::A16;
}


} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearAddRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearAddRoute::A16) {
        nvfp4_linear_add_a16_launch(x, weight, residual, stream);
        return;
    }
    // The dispatcher takes a nullable arena: only the quantized-activation route needs it.
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 linear_add: A4 route requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_linear_add_a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail
