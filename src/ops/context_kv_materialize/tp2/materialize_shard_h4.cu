// The two-device four-KV-head copy of upstream's context_kv_materialize/materialize.cu: the same
// source, compiled with NINFER_CONTEXT_KV_MATERIALIZE_KV_HEADS=4 (a rank's KV heads of the
// DFlash2 drafter: [512,5120] key/value blocks, D128/H4 rings) and its entry point renamed.
// Every kernel, route and epilogue is upstream's; only the row count and the ring's head stride
// change. See docs/maintainer/upstream-merge.md §2.4.
#include "ops/context_kv_materialize/launch.h"
#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/common/warp.cuh"
#include "ops/common/dflash_rope.cuh"

#include "ops/context_kv_materialize/tp2/materialize_h4_names.h"
#include "ops/context_kv_materialize/materialize.cu"
