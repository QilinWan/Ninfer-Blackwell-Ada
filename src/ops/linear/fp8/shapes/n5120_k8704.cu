#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_a8_split_k.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
// Two-device input-column half of [5120,17408]. It inherits the parent's measured schedules and
// A16/A8 route thresholds; they have not been re-measured at this shape.
namespace {
using Geometry = Fp8Geometry<5120, 8704>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Tma32x64 = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using Small = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Mid   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Wide  = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, 170, 4, 8>;
using Bulk  = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<8, 8, 2>>(x, weight, out,
                                                                             stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<64, 2, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 64)
        return launch_fp8_a8_tma<Geometry, Tma32x64>(x, weight, out, scratch, stream);
    if (tokens <= 128)
        return launch_fp8_a8_tma<Geometry, Small>(x, weight, out, scratch, stream);
    if (tokens <= 256)
        return launch_fp8_a8_tma<Geometry, Mid>(x, weight, out, scratch, stream);
    // Wider token tiles avoid an extra wave in the gaps between the bulk anchors.
    if (tokens <= 384 || (tokens > 512 && tokens <= 768))
        return launch_fp8_a8_tma<Geometry, Wide>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // The parent's anchors are wave-boundary choices for these schedules at N=5120, and the halved
    // K keeps the same row tiles, so keep the bands and evaluate the requirement per band: a band
    // reserves partials only for the T values inside it that leave an underfilled final wave.
    const std::size_t last = static_cast<std::size_t>(max_tokens);
    std::size_t bytes      = 0;
    const auto cover       = [&bytes](auto schedule, std::size_t begin, std::size_t end,
                                      std::size_t row_tiles) {
        (void)schedule;
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<decltype(schedule)>(begin, end, row_tiles));
    };
    if (last > 64)
        cover(Small{}, 65, std::min<std::size_t>(last, 128), Geometry::kOutputRows / 128);
    if (last > 128)
        cover(Mid{}, 129, std::min<std::size_t>(last, 256), Geometry::kOutputRows / 128);
    if (last > 256)
        cover(Wide{}, 257, std::min<std::size_t>(last, 384), Geometry::kOutputRows / 128);
    if (last > 384)
        cover(Bulk{}, 385, std::min<std::size_t>(last, 512), Geometry::kOutputRows / 256);
    if (last > 512)
        cover(Wide{}, 513, std::min<std::size_t>(last, 768), Geometry::kOutputRows / 128);
    if (last > 768) cover(Bulk{}, 769, last, Geometry::kOutputRows / 256);
    return bytes;
}

} // namespace

const Fp8LinearShape kFp8N5120K8704{5120,      8704,    launch_a16,
                                    launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
