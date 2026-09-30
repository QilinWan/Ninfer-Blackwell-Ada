#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_split_k.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma96x256 = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

} // namespace

std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t max_tokens) {
    // Every launcher here serves the whole [14336,5120] parent and one device's [7168,5120] shard.
    // Only the bulk route splits K, and the shard has half the parent's 256-row tiles, so it
    // reaches an underfilled final wave from T=289 while the parent stays whole through T=384.
    // The shared capacity therefore reserves the larger of the two requirements.
    const std::size_t last = static_cast<std::size_t>(max_tokens);
    if (last <= 288) return 0;
    const std::size_t parent =
        fp8_a8_band_partial_bytes<Bulk>(289, last, Fp8AttnInputParent::Geometry::kOutputRows / 256);
    const std::size_t shard =
        fp8_a8_band_partial_bytes<Bulk>(289, last, Fp8AttnInputShard::Geometry::kOutputRows / 256);
    return std::max(parent, shard);
}

std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t rows, std::int32_t max_tokens) {
    // Per problem, for the column-parallel wrapper: `fp8_a8_band_partial_bytes` scans the band and
    // answers at the first width that really splits, so the same call gives the parent's T=385
    // start and the shard's T=289 start from its row-tile count alone. The row-count-free query
    // above keeps reserving the larger of the two, so a plan built without `rows` stays safe.
    const std::size_t last = static_cast<std::size_t>(max_tokens);
    if (last <= 288) return 0;
    return fp8_a8_band_partial_bytes<Bulk>(289, last,
                                          static_cast<std::size_t>(rows) / 256);
}

namespace {
template <class Problem>
void run_a8(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& key,
            Tensor& value, Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const typename Problem::Output output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(value.data)};
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, Problem::Geometry::kInputRows>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                      workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    };
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (x.ne[1] <= 96) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= 192) return launch.template operator()<Tma64x256>();
    // Three 96-token tiles give 168 CTAs: one almost-full wave through T=288.
    if (x.ne[1] <= 288) return launch.template operator()<Tma96x256>();
    launch.template operator()<Bulk>();
}
} // namespace

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                              cudaStream_t stream) {
    visit_fp8_attn_input_problem(weight.n, [&]<class Problem>() {
        run_a8<Problem>(x, weight, q, gate, k, v, workspace, stream);
    });
}
} // namespace ninfer::ops::detail
