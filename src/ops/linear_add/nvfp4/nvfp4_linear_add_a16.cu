#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
template <int K, int Ladder = K>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    const int tokens  = x.ne[1];
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto epilogue =
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}};
    if constexpr (Ladder == 6144) {
        if (tokens <= 8) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, K>>(p, output, epilogue,
                                                                           stream);
            return;
        }
        if (tokens <= 16) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, K>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 24) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 1>, K>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, K>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 48) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 2, 2>, K>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 2, 2>, K>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 128) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>, K>>(
                p, output, epilogue, stream);
            return;
        }
        {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
    } else {
        if (tokens <= 8) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 4, 2>, K>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 16) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 4, 2>, K>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 24) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, K>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, K>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 48) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, K>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 2, 2>, K>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 128) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
    }
}
} // namespace

void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 cudaStream_t stream) {
    if (x.ne[1] == 1) {
        nvfp4_linear_add_decode_launch(x, weight, residual, stream);
        return;
    }
        // [5120,3072] is the [5120,6144] family's half, [5120,8704] the [5120,17408] family's.
    if (x.ne[1] <= (weight.k <= 6144 ? 2 : 5)) {
        nvfp4_linear_add_small_t_launch(x, weight, residual, stream);
        return;
    }
    // The halves keep their parent's ladder; the operand's static K is the halved width.
    switch (weight.k) {
    case 6144:
        return launch_matrix<6144>(x, weight, residual, stream);
    case 3072:
        return launch_matrix<3072, 6144>(x, weight, residual, stream);
    case 17408:
        return launch_matrix<17408>(x, weight, residual, stream);
    case 8704:
        return launch_matrix<8704, 17408>(x, weight, residual, stream);
    default:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}
} // namespace ninfer::ops::detail
