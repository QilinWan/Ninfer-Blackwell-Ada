#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_a8_split_k.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
// Two-device input-column half of [5120,6144]. It inherits the parent's measured schedules and
// A16/A8 route thresholds; they have not been re-measured at this shape. Its A8 floor is the fused
// residual projection's (fp8_linear_add_plan.cpp: 17 for the [5120,6144] family), which here equals
// linear()'s own 17: at tp 2, rank 1 runs this linear() while rank 0 runs linear_add over the other
// half of the same row-parallel projection, so the pair takes the same activation route.
// Every K step the parent's ladder uses (GEMV/SIMT 512, sliced-K 256/512, A16 MMA 64/128,
// A8 TMA 128) divides K=3072 = 6 * 512.
namespace {
using Geometry  = Fp8Geometry<5120, 3072>;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Tma32x64  = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
using MidBulk   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 64)
        return launch_fp8_a8_tma<Geometry, Tma32x64>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8<Geometry, Fp8A8T64R64K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    // The narrower row tile fills the GPU before the large-tile path reaches a full wave.
    if (x.ne[1] <= 768)
        return launch_fp8_a8_tma<Geometry, MidBulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // Reserve the largest requirement of every TMA band the interval contains. Only the row-tile
    // count of each schedule matters here, and the halved K does not change it, so this shard
    // keeps its parent's bands; the requirement is evaluated per band rather than by anchor.
    const std::size_t last = static_cast<std::size_t>(max_tokens);
    std::size_t bytes      = 0;
    if (last > 128)
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<Tma64x128>(129, std::min<std::size_t>(last, 192),
                                                                     Geometry::kOutputRows / 128));
    if (last > 192)
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<MidBulk>(193, std::min<std::size_t>(last, 768),
                                                                   Geometry::kOutputRows / 128));
    if (last > 768)
        bytes = std::max(bytes,
                         fp8_a8_band_partial_bytes<Bulk>(769, last, Geometry::kOutputRows / 256));
    return bytes;
}

} // namespace

const Fp8LinearShape kFp8N5120K3072{5120,      3072,    launch_a16,
                                    launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
