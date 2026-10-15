// Translation-unit fragment of src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// The A4 frontier of the snapshot and record forms, for the two-device shard (nvfp4_gdn_input_shard.h).

#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"

namespace ninfer::ops::detail {

bool nvfp4_gdn_conv_uses_a4(LinearPolicy policy, std::int32_t width, std::int32_t batch_size) {
    return nvfp4_gdn_conv_resolve_plan(policy, width, batch_size).schedule ==
               Nvfp4GdnConvScheduleId::Materialized &&
           allows_a4(policy);
}

} // namespace ninfer::ops::detail
