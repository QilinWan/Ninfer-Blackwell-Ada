#pragma once
#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops::detail {
inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

using CausalD256H24Kv4 = CausalGeometry<256, 24, 4>;
using CausalD256H16Kv2 = CausalGeometry<256, 16, 2>;
// The per-device half of 24Q/4KV under two-device tensor parallelism: 12 query heads on 2 KV
// heads, a group of 6. Instantiated by the BF16 and INT8 cache kernels only; the FP8, NVFP4 and
// K8V4 kernels report the geometry as unsupported rather than falling through to 16/2.
using CausalD256H12Kv2 = CausalGeometry<256, 12, 2>;
} // namespace ninfer::ops::detail
