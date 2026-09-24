#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    // The [5120,3072] half takes the warp crossover of the [5120,6144] problem it halves.
    constexpr bool kOutputFamily = Geometry::kInputRows == Nvfp4N5120K6144::kInputRows ||
                                   Geometry::kInputRows == Nvfp4N5120K3072::kInputRows;
    using Schedule = Nvfp4A16SimtSchedule<
        (ActiveTokens <= 16 && ActiveTokens >= (kOutputFamily ? 14 : 8)) ? 16 : 4, 1,
        2, (ActiveTokens >= 17 && ActiveTokens <= 20) ? 8 : 16, ActiveTokens, 1,
        Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default,
        1, Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    launch_nvfp4_a16_simt<
        Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n},
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}}, stream);
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, 2 + static_cast<int>(Offsets)>...};
}

template <class Geometry>
constexpr auto make_launchers() {
    return make_launchers<Geometry>(std::make_index_sequence<5 - 2 + 1>{});
}

constexpr auto kResidual6144Launchers  = make_launchers<Nvfp4N5120K6144>();
constexpr auto kResidual17408Launchers = make_launchers<Nvfp4N5120K17408>();
constexpr auto kResidual8704Launchers  = make_launchers<Nvfp4N5120K8704>();
constexpr auto kResidual3072Launchers  = make_launchers<Nvfp4N5120K3072>();

} // namespace

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - 2);
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        kResidual6144Launchers[index](x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        kResidual17408Launchers[index](x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K8704:
        kResidual8704Launchers[index](x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K3072:
        kResidual3072Launchers[index](x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
    case Nvfp4GeometryId::N17408K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
