// Translation-unit fragment of src/ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp, included once at its end (after its
// closing namespace), so the two-device code below shares that file's private helpers without
// copying them. Not a header. Our file; see docs/maintainer/upstream-merge.md.
//
// The A8 frontier of the snapshot and record forms, for the two-device shard (fp8_gdn_input_shard.h).

#include "ops/gdn_input_proj/tp2/fp8_gdn_input_shard.h"

namespace ninfer::ops::detail {

bool fp8_gdn_conv_uses_a8(LinearPolicy policy, std::int32_t width, std::int32_t batch_size) {
    return fp8_gdn_snapshot_resolve_plan(policy, width, batch_size).schedule ==
           Fp8GdnConvScheduleId::MaterializedA8;
}

} // namespace ninfer::ops::detail
