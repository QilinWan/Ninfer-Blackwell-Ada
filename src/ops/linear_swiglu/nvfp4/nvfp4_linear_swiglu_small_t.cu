#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Geometry is the gate/up problem: the whole [34816,5120] parent, or one device's [17408,5120]
// output-row half. The exact-extent SIMT schedules are the parent's; the half instantiates them at
// its own intermediate row count.
template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    static constexpr int kWarpsPerCta       = ActiveTokens >= 13 ? 16 : (ActiveTokens >= 5 ? 4 : 8);
    using Schedule =
        Nvfp4A16SimtSchedule<kWarpsPerCta, 1, 2, 16, ActiveTokens, 1, kActivationAccess,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    static_assert((kIntermediate % Schedule::kWarpsPerCta) == 0);
    launch_nvfp4_a16_simt<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<1>{});
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, 2 + static_cast<int>(Offsets)>...};
}

template <class Geometry>
constexpr auto kLaunchers = make_launchers<Geometry>(std::make_index_sequence<1>{});

template <class Geometry>
void launch_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    const auto p                = nvfp4_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    if (x.ne[1] <= 2)
        kLaunchers<Geometry>[0](x, weight, out, stream);
    else if (x.ne[1] <= 8)
        launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, Geometry::kInputRows>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
    else
        launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, Geometry::kInputRows>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
}

} // namespace

void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N34816K5120:
        return launch_route<Nvfp4N34816K5120>(x, weight, out, stream);
    case Nvfp4GeometryId::N17408K5120:
        return launch_route<Nvfp4N17408K5120>(x, weight, out, stream);
    default:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_swiglu: unsupported problem");
}

} // namespace ninfer::ops::detail
