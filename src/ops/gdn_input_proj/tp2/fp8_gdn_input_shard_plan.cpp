// The [8192,5120] two-device shard's copy of upstream's fp8/fp8_gdn_input_plan.cpp: the same
// routes and token cutoffs, calling the shard's launchers and split-K reservation
// (fp8_gdn_input_shard.h).
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/tp2/fp8_gdn_input_shard.h"

#include "ops/gdn_input_proj/tp2/fp8_gdn_input_shard_names.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp"
