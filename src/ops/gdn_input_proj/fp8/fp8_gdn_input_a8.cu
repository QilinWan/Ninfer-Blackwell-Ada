#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_split_k.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

template <class Geometry, class Output>
void launch_a8(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
               Fp8A8Workspace workspace, cudaStream_t stream) {
    static_assert(Geometry::kOutputRows == Output::kRows);
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, Geometry::kInputRows>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    };
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= 192) return launch.template operator()<Tma192x128>();
    // Smaller output tiles leave only two full-K tiles to split near the 512-token anchor.
    if (x.ne[1] > 384 && x.ne[1] <= 512) return launch.template operator()<MidBulk>();
    launch.template operator()<Bulk>();
}
} // namespace

std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens) {
    // The shard keeps K, so the schedules are the same, but it halves the row tiles: the same
    // schedule reaches an underfilled final wave in intervals the parent never enters. Reserve the
    // largest requirement of every band the interval contains, for the parent and the shard.
    const std::size_t last = static_cast<std::size_t>(max_tokens);
    std::size_t bytes      = 0;
    const std::size_t parent_rows = Fp8N16384K5120::kOutputRows;
    const std::size_t shard_rows  = Fp8N8192K5120::kOutputRows;
    const auto cover              = [&](auto begin, auto end, auto row_tiles_parent,
                                            auto row_tiles_shard) {
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<MidBulk>(
                                    static_cast<std::size_t>(begin), static_cast<std::size_t>(end),
                                    static_cast<std::size_t>(row_tiles_parent)));
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<MidBulk>(
                                    static_cast<std::size_t>(begin), static_cast<std::size_t>(end),
                                    static_cast<std::size_t>(row_tiles_shard)));
    };
    if (last > 384) cover(385, std::min<std::size_t>(last, 512), parent_rows / 128,
                          shard_rows / 128);
    const auto cover_bulk = [&](auto begin, auto end) {
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<Bulk>(
                                    static_cast<std::size_t>(begin), static_cast<std::size_t>(end),
                                    parent_rows / 256));
        bytes = std::max(bytes, fp8_a8_band_partial_bytes<Bulk>(
                                    static_cast<std::size_t>(begin), static_cast<std::size_t>(end),
                                    shard_rows / 256));
    };
    if (last > 192) cover_bulk(193, std::min<std::size_t>(last, 384));
    if (last > 512) cover_bulk(513, last);
    return bytes;
}

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_a8<Fp8N16384K5120, Fp8GdnInputOutput>(x, weight, qkv, z, workspace, stream);
}

void fp8_gdn_input_shard_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_a8<Fp8N8192K5120, Fp8GdnInputShardOutput>(x, weight, qkv, z, workspace, stream);
}
} // namespace ninfer::ops::detail
