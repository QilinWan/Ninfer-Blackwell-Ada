#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
// Two-device output-row half of the [131072,5120] optimized proposal head (--lm-head-draft with
// MTP at tp 2). It copies the parent's selector (n131072_k5120.cu) and inherits its RTX 5090
// schedules; they have not been re-measured at this shape. Every launcher takes the row count
// from the weight and computes each output row alone, so a half row equals the parent's row bit
// for bit; every block-row tile (4, 32, 64) divides 65536.
Q4Launch select_q4_n65536_k5120(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_k5120_t4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
