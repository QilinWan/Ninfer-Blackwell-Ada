// The [8192,5120] two-device shard's copy of upstream's fp8/fp8_gdn_input_a8.cu: the same
// source, compiled with the shard's section output and entry points (fp8_gdn_input_shard.h).
#include "ops/gdn_input_proj/tp2/fp8_gdn_input_shard_prelude.h"

#include "ops/gdn_input_proj/tp2/fp8_gdn_input_shard_names.h"
// This source also defines the parent's partial reservation; its copy here is left unused.
#undef fp8_gdn_input_partial_capacity_bytes
#define fp8_gdn_input_partial_capacity_bytes fp8_gdn_input_parent_partial_capacity_bytes_unused
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu"
#undef fp8_gdn_input_partial_capacity_bytes

namespace ninfer::ops::detail {

// The shard's own split-K reservation: 32 row tiles, so Bulk (upstream's schedule above) splits
// its last wave from T=193 (64 tiles) on, where the parent splits from T=257; MidBulk needs less.
std::size_t fp8_gdn_input_shard_partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 192 ? Bulk::kPartialBytes : 0;
}

} // namespace ninfer::ops::detail
