// The two-device [17408,5120] half's copy of upstream's fp8/fp8_linear_swiglu_a16.cu: the same
// source, compiled with the half's geometry and entry points (linear_swiglu_half.h).
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/tp2/linear_swiglu_half.h"

#include "ops/linear_swiglu/tp2/fp8_linear_swiglu_half_names.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_a16.cu"
