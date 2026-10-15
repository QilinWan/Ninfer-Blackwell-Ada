# Ada (sm_89) compatibility layer.
#
# Upstream qualifies exactly one GPU architecture, sm_120a (Blackwell). A few
# hundred CUDA lines encode instructions that do not exist on Ada:
#
#   * FP4 code conversion (`cvt.rn.satfinite.e2m1x2.f32`) and block-scaled MMA
#     (`mma...kind::mxf8f6f4...scale::`) - NVFP4 and K8V4 weight/KV routes,
#   * TMA (`cp.async.bulk.tensor`) plus `setmaxnreg` warp-specialized kernels in
#     the separate non-RDC archive,
#   * `.op_restrict` and programmatic-dependent-launch modifiers gated to sm_90+.
#
# The Ada build therefore drops those translation units from the archives and
# links a stub that answers the removed routes with an explicit unsupported
# status. Every remaining route - groupwise INT (Q4/Q5/Q6/Q8), FP8 A8/A16, BF16,
# INT8 and FP8 KV - compiles and runs on Ada. Included from src/ops/CMakeLists.txt.

if(NOT NINFER_SM89_BUILD)
  return()
endif()

set(NINFER_SM89_DROPPED_PATTERNS
  "/nvfp4/.*w4a4.*\\.cu$"
  "/nvfp4/shapes/.*\\.cu$"
  "/nvfp4_w4a4\\.cu$"
  "/nvfp4_w4a4_tma\\.cu$"
  "/kv_cache/append/(nvfp4|k8v4)_launch\\.cu$"
  "/causal_cache/(small_t|prompt)_(nvfp4|k8v4)\\.cu$"
)

get_target_property(_ninfer_ops_sources ninfer_ops SOURCES)
set(_ninfer_dropped "")
foreach(_pattern IN LISTS NINFER_SM89_DROPPED_PATTERNS)
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX "${_pattern}")
endforeach()
set_property(TARGET ninfer_ops PROPERTY SOURCES ${_ninfer_ops_sources})

target_sources(ninfer_ops PRIVATE "${PROJECT_SOURCE_DIR}/src/ops/sm89/nvfp4_sm89_stubs.cpp")

# The non-RDC archive exists only for Blackwell warp-specialized kernels. Keep the
# target name so family manifests can keep linking into it, but leave it empty of
# device code on Ada.
get_target_property(_ninfer_non_rdc_sources ninfer_nvfp4_non_rdc SOURCES)
list(LENGTH _ninfer_non_rdc_sources _ninfer_non_rdc_count)
set_property(TARGET ninfer_nvfp4_non_rdc PROPERTY SOURCES)
target_sources(ninfer_nvfp4_non_rdc
  PRIVATE "${PROJECT_SOURCE_DIR}/src/ops/sm89/nvfp4_non_rdc_sm89_empty.cpp")
message(STATUS "sm_89 build: dropped ${_ninfer_non_rdc_count} Blackwell-only non-RDC "
               "translation units and the NVFP4/K8V4 kernel routes")
