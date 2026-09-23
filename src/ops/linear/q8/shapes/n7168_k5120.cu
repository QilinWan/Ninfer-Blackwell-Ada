#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
// Two-device tensor-parallel half: [7168,5120] from [14336,5120] — the packed MTP attention input,
// output rows halved. It inherits the parent's measured A16 sliced-K ladder and large-T tail
// verbatim: `with_problem<Geometry::kInputRows, Capacity>` binds the operands' static
// K to this shape's own extent, so the parent's measured tiles stay well-formed. Not
// re-measured at this shape. Q8 reduces K-slice partials in shared memory, so the
// caller-side capacity quote stays 0 (linear_workspace_capacity_bytes), and both
// row-parallel ranks take identical activation routes: this family has no A8 route.
namespace {
using Geometry = Q8N7168K5120;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C4 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 3, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8 =
    Q8A16SlicedKMmaSchedule<8, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
// Bound registers to allow six resident 128-thread CTAs per SM.
using C16 = Q8A16SlicedKMmaSchedule<16, 4, 1, 6, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C24 = Q8A16SlicedKMmaSchedule<24, 8, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C48 = Q8A16SlicedKMmaSchedule<48, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C56 = Q8A16SlicedKMmaSchedule<56, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
} // namespace

Q8Launch select_q8_n7168_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
