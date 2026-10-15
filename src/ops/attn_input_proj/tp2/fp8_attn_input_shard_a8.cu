// The [7168,5120] two-device shard's copy of upstream's fp8/fp8_attn_input_a8.cu: the same
// source, compiled with the shard's section output and entry points (fp8_attn_input_shard.h).
#include "ops/attn_input_proj/tp2/fp8_attn_input_shard_prelude.h"

#include "ops/attn_input_proj/tp2/fp8_attn_input_shard_names.h"
// This source also defines the parent's partial reservation; its copy here is left unused.
#undef fp8_attn_input_partial_capacity_bytes
#define fp8_attn_input_partial_capacity_bytes fp8_attn_input_parent_partial_capacity_bytes_unused
#include "ops/attn_input_proj/fp8/fp8_attn_input_a8.cu"
#undef fp8_attn_input_partial_capacity_bytes

namespace ninfer::ops::detail {

// The shard's own split-K reservation: 28 row tiles, so Bulk (upstream's schedule above) splits
// its last wave from T=289 (84 tiles) on, where the parent's 56 tiles split from T=385.
std::size_t fp8_attn_input_shard_partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 288 ? Bulk::kPartialBytes : 0;
}

} // namespace ninfer::ops::detail
