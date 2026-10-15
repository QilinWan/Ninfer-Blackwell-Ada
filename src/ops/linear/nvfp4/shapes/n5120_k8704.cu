#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
// Two-device input-column half of [5120,17408]. It inherits the parent's measured schedules and
// route thresholds (n5120_k17408.cu); they have not been re-measured at this shape. Its A4 floor is
// the MLP down family's constant, which the fused residual projection (nvfp4_linear_add_plan.cpp)
// reads too: at tp 2 rank 1 runs this linear() and rank 0 runs linear_add over the other half of
// the same row-parallel projection, so both ranks take the same activation route at every width.
// 8704 = 17 * 512 = 34 * 256 = 68 K128 tiles, so every parent instance keeps whole K tiles.
namespace {
using Geometry = Nvfp4Geometry<5120, 8704>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C2      = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SimtBlockOrder::RowsContiguous, 1>;
// C4 (T=3..4) and C5 ask for 6 CTAs per SM (<= 80 registers instead of 112). Four rows per warp
// (16 per CTA, 320 CTAs) instead of two: each warp amortizes its activation reads over twice the
// weight rows, and the per-row K loop is unchanged, so outputs stay bit-identical. Measured with
// ninfer_linear_bench on an RTX 5070 Ti (70 SMs), medians of 3x60: T=3/4 46.8 -> 44.7 us
// (-4.5 %), T=5 53.0 -> 46.8 us (-11.6 %). Two rows per warp was the previous choice (and the
// parent's); one row per warp costs +30 %, eight rows spill registers (+120 %).
using C4      = Nvfp4A16SimtSchedule<4, 1, 4, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SimtBlockOrder::RowsContiguous, 6>;
using C5      = Nvfp4A16SimtSchedule<4, 1, 4, 16, 5, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SimtBlockOrder::RowsContiguous, 6>;
using T32R64  = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128 = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using T128R128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
    if (tokens == 5) return nvfp4_linear_a16_simt<Geometry, 5, C5, true>;
    // Eight K-warps (256 threads) instead of four at T=6..8: 53.0 -> 50.9 us (-3.9 %) on the
    // RTX 5070 Ti; the K partition changes, so T=6..8 outputs are not bit-identical to the parent.
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 4, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 48) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 1024) return nvfp4_a4_tma_route<Nvfp4GeometryId::N5120K8704>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) {
    return max_tokens >= kNvfp4DownFamilyFirstA4Tokens;
}
} // namespace

const Nvfp4LinearShape kNvfp4N5120K8704{5120, 8704, launch_a16, launch_nvfp4_a4<select_a4>,
                                        uses_a4};
} // namespace ninfer::ops::detail
