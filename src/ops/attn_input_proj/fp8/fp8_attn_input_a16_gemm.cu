#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/common/token_slices.h"
#include "core/device.h"

namespace ninfer::ops::detail {
namespace {
template <class Problem, class S>
void run(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
         cudaStream_t stream) {
    using Output = typename Problem::Output;
    static_assert(Output::kQueryRows % S::kBlockRows == 0 && Output::kKeyRows % S::kBlockRows == 0);
    const typename Problem::Output output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_fp8_a16_mma<Fp8ScheduleInstance<S, Problem::Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <class Problem>
void run_problem(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                 Tensor& v, cudaStream_t stream) {
    // Column tiles follow the measured whole-Op envelope, including the 129..160 tail.
    if (x.ne[1] <= 64)
        run<Problem, Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, q, gate, k, v,
                                                                   stream);
    else if (x.ne[1] <= 80 || (x.ne[1] >= 129 && x.ne[1] <= 160))
        run<Problem, Fp8A16MmaSchedule<32, 80, 128, 32, 16, 1, 3>>(x, weight, q, gate, k, v,
                                                                   stream);
    else if (x.ne[1] <= 96)
        run<Problem, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, q, gate, k, v,
                                                                   stream);
    else
        run<Problem, Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, q, gate, k, v,
                                                                   stream);
}
} // namespace

void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    visit_fp8_attn_input_problem(
        weight.n,
        [&]<class Problem>() { run_problem<Problem>(x, weight, q, gate, k, v, stream); });
}
} // namespace ninfer::ops::detail
