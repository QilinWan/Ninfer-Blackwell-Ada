#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Column tiles amortize gate/up decode over the complete speculative block.
using M32N128  = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128  = Nvfp4A4MmaSchedule<64, 128, 256, 4, 4, 2, 1>;
using M128N128 = Nvfp4A4MmaSchedule<128, 128, 256, 4, 4, 2, 1>;
using M96N128  = Nvfp4A4MmaSchedule<96, 128, 256, 3, 4, 2, 1>;

// Geometry is the gate/up problem: the whole [34816,5120] parent, or one device's [17408,5120]
// output-row half under two-device tensor parallelism. Both share K and the fused-to-intermediate
// ratio, so the schedules, the layout and the token cutoffs are the parent's; only the paired
// gate/up row span, which the row policy reads from the operand rows, differs.
template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4A4Workspace workspace, int tokens,
                 cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a4_mma<S>(nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
                           SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}

template <class Geometry, class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
            cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4A4Workspace scratch =
        allocate_nvfp4_a4_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::RowMajor, stream);
    launch_gemm<Geometry, Schedule>(weight, out, scratch, x.ne[1], stream);
}

template <class Geometry>
void launch_route(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    if (x.ne[1] <= 32) {
        launch<Geometry, M32N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M64N128::kBlockTokens) {
        launch<Geometry, M64N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M96N128::kBlockTokens) {
        launch<Geometry, M96N128>(x, weight, out, workspace, stream);
    } else {
        launch<Geometry, M128N128>(x, weight, out, workspace, stream);
    }
}

} // namespace

void nvfp4_linear_swiglu_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N34816K5120:
        return launch_route<Nvfp4N34816K5120>(x, weight, out, workspace, stream);
    case Nvfp4GeometryId::N17408K5120:
        return launch_route<Nvfp4N17408K5120>(x, weight, out, workspace, stream);
    default:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_swiglu: unsupported problem");
}

} // namespace ninfer::ops::detail
