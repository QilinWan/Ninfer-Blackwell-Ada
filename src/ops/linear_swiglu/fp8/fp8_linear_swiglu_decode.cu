#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Schedule = Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                    Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
static_assert(Schedule::kRowsPerWarp == 2);

// The whole [34816,5120] parent and one device's [17408,5120] output-row half: same K, same
// schedule and 4-token capacity, different compile-time intermediate row count.
template <class Geometry>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    static_assert((kIntermediate % Schedule::kWarpsPerCta) == 0);
    using Rows = Fp8SwiGluRows<Schedule::kRowsPerWarp / 2, kIntermediate>;
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] != 1 || out.ne[0] != kIntermediate ||
        out.ne[1] != 1 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("fp8 linear_swiglu decode: invalid exact problem");
    }
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, 4>>(
        fp8_a16_operands(x, weight), output, Fp8SwiGluEpilogue{}, stream, Rows{});
}

} // namespace

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N34816K5120:
        return launch_decode<Fp8N34816K5120>(x, weight, out, stream);
    case Fp8GeometryId::N17408K5120:
        return launch_decode<Fp8N17408K5120>(x, weight, out, stream);
    default:
        break;
    }
    throw std::invalid_argument("fp8 linear_swiglu: unsupported problem");
}

} // namespace ninfer::ops::detail
