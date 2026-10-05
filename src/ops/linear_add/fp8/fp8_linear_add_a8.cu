#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
namespace {
// The attention output family: [5120,6144] and its two-device input-column half [5120,3072]. The
// other family is the MLP down projection [5120,17408] and its half [5120,8704]. Each half runs
// the schedules and partial reservation of the problem it halves (not re-measured at the half K);
// every row and K tile below divides 5120, 3072 and 8704.
constexpr bool output_family(int k) { return k == 6144 || k == 3072; }

// This Op admits dense, even-width BF16 residuals. Adjacent MMA rows share one load. The scaled
// update is one FMA in every tile: the full 1024-token prefill chunks already contracted it (the
// FullTokens instance and the split-K reduce), only tail tiles rounded the sum separately.
struct Fp8ResidualAddEpilogue : LinearResidualAddEpilogue {
    __device__ __forceinline__ float apply_scaled(int row, int token, float value,
                                                  float scale) const {
        return __fmaf_rn(value, scale, residual.load(row, token));
    }

    __device__ __forceinline__ float2 apply_row_pair_scaled(int row, int token, float2 value,
                                                            float2 scale) const {
        const auto* pointer =
            residual.data + static_cast<std::int64_t>(token) * residual.leading_dim + row;
        const float2 add = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pointer));
        return make_float2(__fmaf_rn(value.x, scale.x, add.x), __fmaf_rn(value.y, scale.y, add.y));
    }
};

using K6144Tma32x64  = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using K6144Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
using K6144MidBulk =
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using K6144Bulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

using K17408Tma32x64 = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using K17408Small =
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using K17408Mid = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using K17408Wide =
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, 170, 4, 8>;
using K17408Bulk =
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

template <int K>
void launch_problem(const Tensor& x, const Weight& weight, Tensor& residual,
                    Fp8A8Workspace workspace, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output output{data, weight.n};
    const Fp8ResidualAddEpilogue epilogue{{{data, weight.n}}};
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, K>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, epilogue, stream, workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, epilogue, stream);
    };
    // K is the contraction width; the two-device input-column halves [5120,3072] and [5120,8704]
    // run their parent's ladder (N and the row-tile count are unchanged, and every tile divides the
    // halved K), so each half maps to the family it halves.
    if constexpr (K == 6144 || K == 3072) {
        if (x.ne[1] <= 64) return launch.template operator()<K6144Tma32x64>();
        if (x.ne[1] <= 128) return launch.template operator()<Fp8A8T64R64K128>();
        if (x.ne[1] <= 192) return launch.template operator()<K6144Tma64x128>();
        // The narrower row tile fills the GPU before the large-tile path reaches a full wave.
        if (x.ne[1] <= 768) return launch.template operator()<K6144MidBulk>();
        launch.template operator()<K6144Bulk>();
    } else {
        const int tokens = x.ne[1];
        if (tokens <= 64) return launch.template operator()<K17408Tma32x64>();
        if (tokens <= 128) return launch.template operator()<K17408Small>();
        if (tokens <= 256) return launch.template operator()<K17408Mid>();
        // Wider token tiles avoid an extra wave in the gaps between the bulk anchors.
        if (tokens <= 384 || (tokens > 512 && tokens <= 768))
            return launch.template operator()<K17408Wide>();
        launch.template operator()<K17408Bulk>();
    }
}
} // namespace

std::size_t fp8_linear_add_partial_capacity_bytes(std::int32_t k, std::int32_t max_tokens) {
    if (k == 6144 || k == 3072) {
        if (max_tokens > 768) return K6144Bulk::kPartialBytes;
        return max_tokens > 192 ? K6144MidBulk::kPartialBytes : 0;
    }
    if (max_tokens > 384) return K17408Bulk::kPartialBytes;
    if (max_tokens > 256) return K17408Wide::kPartialBytes;
    if (max_tokens > 128) return K17408Mid::kPartialBytes;
    return max_tokens > 64 ? K17408Small::kPartialBytes : 0;
}

void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_linear_add_partial_capacity_bytes(weight.k, x.ne[1]));
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    // The A8 route is keyed on the contraction width: the two-device input-column halves keep the
    // ladder and the wave anchors of the problem they halve (N=5120 is unchanged).
    switch (weight.k) {
    case Fp8N5120K6144::kInputRows:
        return launch_problem<Fp8N5120K6144::kInputRows>(x, weight, residual, scratch, stream);
    case Fp8N5120K3072::kInputRows:
        return launch_problem<Fp8N5120K3072::kInputRows>(x, weight, residual, scratch, stream);
    case Fp8N5120K17408::kInputRows:
        return launch_problem<Fp8N5120K17408::kInputRows>(x, weight, residual, scratch, stream);
    case Fp8N5120K8704::kInputRows:
        return launch_problem<Fp8N5120K8704::kInputRows>(x, weight, residual, scratch, stream);
    default:
        break;
    }
    throw std::invalid_argument("fp8 linear_add: unsupported problem");
}
} // namespace ninfer::ops::detail
