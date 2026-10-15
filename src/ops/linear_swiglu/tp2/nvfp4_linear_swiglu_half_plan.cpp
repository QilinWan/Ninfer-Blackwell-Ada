// The two-device [17408,5120] half's copy of upstream's nvfp4/nvfp4_linear_swiglu_plan.cpp: the same
// source, compiled with the half's geometry and entry points (linear_swiglu_half.h).
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/tp2/linear_swiglu_half.h"

namespace ninfer::ops::detail {
static_assert(Nvfp4N17408K5120::kOutputRows == kLinearSwiGluHalfRows &&
              Nvfp4N17408K5120::kInputRows == Nvfp4N34816K5120::kInputRows);
} // namespace ninfer::ops::detail

#include "ops/linear_swiglu/tp2/nvfp4_linear_swiglu_half_names.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp"
