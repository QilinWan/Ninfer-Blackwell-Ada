#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"

namespace ninfer::ops::detail {
namespace {

// One A16 ladder serves the [14336,5120] parent and a rank's [7168,5120] shard: the shard keeps K,
// so every schedule instance is the parent's and only `Output`'s section rows differ. The measured
// crossovers are the parent's, which is also what the workspace query reports.
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
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 32) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 64) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 128) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    launch_nvfp4_a16_mma<
        Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 5120>>(
        p, output, epilogue, stream);
}

} // namespace

void nvfp4_attn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        nvfp4_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (tokens <= 2) {
        nvfp4_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (weight.n == kNvfp4AttnInputShardRows) {
        launch_a16_ladder<Nvfp4AttentionInputShardOutput>(
            x, weight,
            {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
             static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
            stream);
        return;
    }
    launch_a16_ladder<Nvfp4AttentionInputOutput>(
        x, weight,
        {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
         static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
        stream);
}
} // namespace ninfer::ops::detail
