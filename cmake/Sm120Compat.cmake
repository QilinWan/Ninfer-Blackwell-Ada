# Blackwell (sm_120a) side of the E8 KV route split.
#
# The rk4v4-e8 storage mode (E8 Conway-Sloane lattice keys, i4x16 values) was an Ada-era codec
# kept for the removed sm_89 build. This fork builds only Blackwell, which ships NVFP4, so the
# E8 kernel translation units are dropped from the archives and a stub answers the removed routes
# with an explicit unsupported status. Included from src/ops/CMakeLists.txt.

set(NINFER_SM120A_DROPPED_PATTERNS
  "/kv_cache/append/e8_launch\\.cu$"
  "/causal_cache/e8/.*\\.(cu|cpp)$"
)

get_target_property(_ninfer_ops_sources ninfer_ops SOURCES)
foreach(_pattern IN LISTS NINFER_SM120A_DROPPED_PATTERNS)
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX "${_pattern}")
endforeach()
set_property(TARGET ninfer_ops PROPERTY SOURCES ${_ninfer_ops_sources})

target_sources(ninfer_ops PRIVATE "${PROJECT_SOURCE_DIR}/src/ops/sm120a/e8_sm120a_stubs.cpp")

message(STATUS "sm_120a build: dropped the Ada-only E8 KV append route; "
               "Blackwell uses the native NVFP4 KV cache")
