#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
// Two-device input-column half of [5120,6144]. It inherits the parent's measured schedules and
// token cutoffs; they have not been re-measured at this shape. Every K step of those schedules
// (GEMV/SIMT phase 512/256, sliced-K 256, MMA K 192/128/64) divides K=3072.
namespace {
using Gemv = Bf16A16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
using C2   = Bf16A16SimtSchedule<4, 1, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C4   = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
} // namespace

Bf16Launch select_bf16_n5120_k3072(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 3072>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<C2, 3072, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<C4, 3072, 4>>;
    if (tokens <= 32)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 3072>>;
    if (tokens <= 64) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K192S2, 3072>>;
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 3072>>;
    if (tokens <= 192)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 3072>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 3072>>;
}
} // namespace ninfer::ops::detail
