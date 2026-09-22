#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Problem>
void run_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    using Geometry = typename Problem::Geometry;
    // Keep six CTAs resident with the four-segment output (39 registers, no spills).
    using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 6>;
    const typename Problem::Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

} // namespace

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream) {
    visit_fp8_attn_input_problem(weight.n, [&]<class Problem>() {
        run_decode<Problem>(x, weight, q, gate, k, v, stream);
    });
}

} // namespace ninfer::ops::detail
