#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"

namespace ninfer::ops::detail {
namespace {

// K is the parent's in both problems, so the schedule instances and the quantized operands are
// shared; only the section rows of the output differ (nvfp4_gdn_input_output.cuh).
using Geometry = Nvfp4N16384K5120;

using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

// This projection selects its own route, so the layout the quantizer writes below must be derived
// from the same predicate; the two are read together at the call site for that reason.
constexpr bool uses_tma(std::int32_t tokens) { return tokens >= 512; }

template <class Schedule, class Output>
void launch_gemm(const Weight& weight, const Output& output, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor), output,
        LinearIdentityEpilogue{}, stream);
}

template <class Output>
void launch_gemm_ladder(const Weight& weight, const Output& output, std::int32_t tokens,
                        Nvfp4A4Workspace workspace, cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<M32N64>(weight, output, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<M32N128>(weight, output, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<M128N128Pipelined>(weight, output, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<M64N128>(weight, output, workspace, tokens, stream);
    } else {
        launch_gemm<M128N128Resident>(weight, output, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_gdn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const auto layout = uses_tma(tokens) ? Nvfp4ScaleLayout::Tiled256 : Nvfp4ScaleLayout::RowMajor;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    if (uses_tma(tokens)) {
        launch_nvfp4_a4_tma_gdn(nvfp4_a4_operands(weight, workspace, tokens, layout),
                                static_cast<__nv_bfloat16*>(qkv.data),
                                static_cast<__nv_bfloat16*>(z.data), stream);
        return;
    }
    if (weight.n == kNvfp4GdnInputShardRows) {
        launch_gemm_ladder<Nvfp4GdnInputShardOutput>(
            weight,
            {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}, tokens,
            workspace, stream);
        return;
    }
    launch_gemm_ladder<Nvfp4GdnInputOutput>(
        weight, {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}, tokens,
        workspace, stream);
}

} // namespace ninfer::ops::detail
