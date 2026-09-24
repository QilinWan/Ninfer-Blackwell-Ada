#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"

namespace ninfer::ops::detail {
namespace {

// One A16 ladder serves the [16384,5120] parent and a rank's [8192,5120] shard: the shard keeps K,
// so every schedule instance is the parent's and only `Output`'s section rows differ. That is also
// why the workspace query reports one capacity for both.
template <class Output>
void launch_a16_ladder(const Tensor& x, const Weight& weight, const Output& output,
                       cudaStream_t stream) {
    const int tokens    = x.ne[1];
    const auto p        = nvfp4_a16_operands(x, weight);
    const auto epilogue = LinearIdentityEpilogue{};
    if (tokens <= 8) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 16) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 24) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 1>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 32) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 64) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 96) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    launch_nvfp4_a16_mma<
        Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 5120>>(
        p, output, epilogue, stream);
}

} // namespace

void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        nvfp4_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    }
    if (tokens <= 2) {
        nvfp4_gdn_input_small_t_launch(x, weight, qkv, z, stream);
        return;
    }
    if (weight.n == kNvfp4GdnInputShardRows) {
        launch_a16_ladder<Nvfp4GdnInputShardOutput>(
            x, weight,
            {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}, stream);
        return;
    }
    launch_a16_ladder<Nvfp4GdnInputOutput>(
        x, weight, {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        stream);
}
} // namespace ninfer::ops::detail
