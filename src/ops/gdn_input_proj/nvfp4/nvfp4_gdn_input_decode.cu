#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

namespace ninfer::ops::detail {
namespace {

// One decode launcher serves the [16384,5120] parent and a rank's [8192,5120] shard: the shard keeps
// K, so only the geometry and the section rows of the output differ.
template <class Geometry, class Output>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    static_assert((Geometry::kOutputRows % 128) == 0);

    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight),
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

} // namespace

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    if (weight.n == kNvfp4GdnInputShardRows) {
        launch<Nvfp4N8192K5120, Nvfp4GdnInputShardOutput>(x, weight, qkv, z, stream);
        return;
    }
    launch<Nvfp4N16384K5120, Nvfp4GdnInputOutput>(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
