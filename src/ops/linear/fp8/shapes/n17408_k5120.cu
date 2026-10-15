#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_a8_split_k.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
// Two-device output-row half of [34816,5120]. It inherits the parent's measured schedules and
// A16/A8 route thresholds; they have not been re-measured at this shape.
namespace {
using Geometry   = Fp8Geometry<17408, 5120>;
using C4         = Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                      Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256  = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma128x256 = Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>;
using Bulk       = Fp8A8TmaSplitKSchedule<Tma128x256, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 4) return fp8_linear_a16_simt<Geometry, 4, C4>(x, weight, out, stream);
    if (tokens <= 8)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<8, 4, 1>>(x, weight, out,
                                                                             stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 2, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(
            x, weight, out, stream);
    if (tokens <= 96)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 64)
        return launch_fp8_a8<Geometry, Fp8A8T64R128K256>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma64x256>(x, weight, out, scratch, stream);
    // At T <= 256 the last wave is already well filled, so Bulk uses its ordinary
    // TMA kernel and needs no partials. Keep one compiled family for this region.
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 5; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // Reserve the largest requirement of every TMA band the interval contains; the shard's row
    // tiles are half the parent's, so the parent's 256-token anchor sits inside a splitting band.
    const std::size_t last = static_cast<std::size_t>(max_tokens);
    std::size_t bytes      = 0;
    if (last > 64)
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<Tma64x128>(65, std::min<std::size_t>(last, 128),
                                                                     Geometry::kOutputRows / 128));
    if (last > 128)
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<Tma64x256>(129, std::min<std::size_t>(last, 192),
                                                                     Geometry::kOutputRows / 256));
    if (last > 192)
        bytes = std::max(bytes,
                         fp8_a8_band_partial_bytes<Bulk>(193, last, Geometry::kOutputRows / 256));
    return bytes;
}
} // namespace

const Fp8LinearShape kFp8N17408K5120{17408,     5120,    launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
