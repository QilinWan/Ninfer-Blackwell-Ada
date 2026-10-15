find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(ops/tests.cmake)

if(NINFER_SM89_BUILD)
  # These cases drive FP4 code conversion, block-scaled MMA and TMA kernels that an Ada build does
  # not contain at all (cmake/Sm89Compat.cmake). Disabling them keeps `ctest` a meaningful gate on
  # sm_89 instead of turning an absent architecture feature into a failure. The mixed Op binaries
  # keep running: their FP4 case blocks are guarded by NINFER_SM89 at the call site.
  set(_ninfer_sm89_disabled
    ninfer_linear_nvfp4_a16_test
    ninfer_linear_nvfp4_a4_test
    ninfer_linear_add_nvfp4_test
    ninfer_linear_swiglu_nvfp4_test
    ninfer_softmax_attention_nvfp4_test
    ninfer_softmax_attention_k8v4_test
    ninfer_kv_cache_append_nvfp4_test
    ninfer_kv_cache_append_k8v4_test)
  foreach(_ninfer_test IN LISTS _ninfer_sm89_disabled)
    if(TEST ${_ninfer_test})
      set_tests_properties(${_ninfer_test} PROPERTIES DISABLED TRUE)
    else()
      message(FATAL_ERROR "sm_89 test gate names an unregistered test: ${_ninfer_test}")
    endif()
  endforeach()
endif()
