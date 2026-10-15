// The two-device [3072,5120] shard's dispatch: the parent's DFlash2 route table
// (q8/q8_attn_input_plan.cpp, kDFlash2Routes) over the shard's launchers. Our file.
#include "ops/attn_input_proj/tp2/q8_dflash2_attn_input_shard.h"
#include "ninfer/ops/tp2/attn_input_proj.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void q8_dflash2_attn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (weight.qtype != QType::Q8_G32_FP16 || weight.n != kQ8DFlash2ShardRows ||
        weight.k != 5120 || weight.padded_shape[1] != 5120 || x.ne[0] != 5120 || cols < 1 ||
        q.ne[0] != kQ8DFlash2ShardQueryRows || k.ne[0] != kQ8DFlash2ShardKvRows ||
        v.ne[0] != kQ8DFlash2ShardKvRows || q.ne[1] != cols || k.ne[1] != cols ||
        v.ne[1] != cols) {
        throw std::invalid_argument("Q8 DFlash2 attention input shard: invalid problem");
    }
    if (cols <= 48) return q8_dflash2_attn_input_shard_small_t_launch(x, weight, q, k, v, stream);
    if (cols <= 63)
        return q8_dflash2_attn_input_shard_mma_r16_c64_k128_launch(x, weight, q, k, v, stream);
    if (cols <= 96)
        return q8_dflash2_attn_input_shard_mma_r32_c32_k128_launch(x, weight, q, k, v, stream);
    if (cols <= 128)
        return q8_dflash2_attn_input_shard_mma_r32_c64_k128_launch(x, weight, q, k, v, stream);
    if (cols <= 192)
        return q8_dflash2_attn_input_shard_mma_r32_c64_launch(x, weight, q, k, v, stream);
    q8_dflash2_attn_input_shard_mma_r64_c128_launch(x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail

namespace ninfer::ops {

void attn_input_proj_head_block(const Tensor& x, const Weight& query_key_value_weight, Tensor& q,
                                Tensor& k, Tensor& v, cudaStream_t stream) {
    const auto contiguous_bf16 = [](const Tensor& t) {
        return t.dtype == DType::BF16 && t.is_contiguous() && t.data != nullptr && t.ne[2] == 1 &&
               t.ne[3] == 1;
    };
    if (!contiguous_bf16(x) || !contiguous_bf16(q) || !contiguous_bf16(k) || !contiguous_bf16(v) ||
        query_key_value_weight.layout != QuantLayout::RowSplit ||
        query_key_value_weight.qdata == nullptr || query_key_value_weight.scales == nullptr) {
        throw std::invalid_argument("attn_input_proj_head_block: invalid operands");
    }
    detail::q8_dflash2_attn_input_shard_dispatch(x, query_key_value_weight, q, k, v, stream);
}

} // namespace ninfer::ops
