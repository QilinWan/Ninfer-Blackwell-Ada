#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q4Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q4_n1024_k5120},
    ShapeEntry{4096, 5120, select_q4_n4096_k5120},
    ShapeEntry{5120, 6144, select_q4_n5120_k6144},
    ShapeEntry{6144, 5120, select_q4_n6144_k5120},
    ShapeEntry{7168, 5120, select_q4_n7168_k5120},
    ShapeEntry{34816, 5120, select_q4_n34816_k5120},
    ShapeEntry{131072, 5120, select_q4_n131072_k5120},
    ShapeEntry{131072, 2048, select_q4_n131072_k2048},
    ShapeEntry{3456, 1152, select_q4_n3456_k1152},
    ShapeEntry{4304, 1152, select_q4_n4304_k1152},
};
} // namespace

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    // Unregistered shapes (Qwen3.5-9B: K=4096/12288) take generic routes, untuned. The
    // exact-T and draft-head launchers stay behind their registered entries.
    if (k % 128 == 0 && n % 64 == 0) {
        if (t == 1) { return n == 131072 ? launch_q4_gemv_r4_w1_direct : launch_q4_gemv_r1_q8_direct; }
        if (t <= 4) { return launch_q4_simt_r8_c4; }
        if (t <= 16) { return launch_q4_simt_r8_c8; }
        return launch_q4_mma_r64_c128;
    }
    throw std::invalid_argument("q4 linear: unsupported shape");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q4 linear: unsupported policy");
    return select_q4_a16_launch(n, k, t);
}

void q4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q4_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
