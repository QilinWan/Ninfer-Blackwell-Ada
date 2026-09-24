#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

// One decode launcher serves the [14336,5120] parent and a rank's [7168,5120] shard: the shard keeps
// K, so only the geometry and the section rows of the output differ.
template <class Geometry, class Output>
void launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
            cudaStream_t stream) {
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    static_assert((Geometry::kOutputRows % 128) == 0);

    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

} // namespace

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    if (weight.n == kNvfp4AttnInputShardRows) {
        launch<Nvfp4N7168K5120, Nvfp4AttentionInputShardOutput>(x, weight, q, gate, k, v, stream);
        return;
    }
    launch<Nvfp4N14336K5120, Nvfp4AttentionInputOutput>(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
