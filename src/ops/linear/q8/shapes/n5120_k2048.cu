#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
// Two-device input-column half of the DFlash2 drafter's MLP down projection [5120,4096] — a rank's
// 2048 intermediate columns. The whole [5120,4096] parent runs inside
// linear_dynamic_grouped_conv_add, not as a linear shape, so this inherits the measured A16
// sliced-K ladder of the MTP half (n5120_k3072.cu) and is not re-measured here. 2048 = 4 * 512,
// so every sliced-K instance keeps whole K slices (KWarps 4/8 = 256/512), and Q8 reduces
// K-slice partials in shared memory (caller-side capacity quote stays 0).
namespace {
using Geometry = Q8N5120K2048;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C4 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16 =
    Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C24 =
    Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C32 = Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C40 =
    Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C48 =
    Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C56 =
    Q8A16SlicedKMmaSchedule<56, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C64 =
    Q8A16SlicedKMmaSchedule<64, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
} // namespace

Q8Launch select_q8_n5120_k2048(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_a16_sliced<Geometry, 64, C64>;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
