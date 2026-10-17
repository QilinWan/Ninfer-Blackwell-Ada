find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(ops/tests.cmake)

if(NOT NINFER_ENABLE_VISION)
  # A text-only build links the media decoder stand-in by design, so the cases that decode image or
  # video input cannot run in it. Disabling them keeps the suite a real gate for both configurations.
  set(_ninfer_visionless_tests
    ninfer_media_decode_test
    ninfer_qwen3_5_frontend_test)
  foreach(_ninfer_test IN LISTS _ninfer_visionless_tests)
    if(TEST ${_ninfer_test})
      set_tests_properties(${_ninfer_test} PROPERTIES DISABLED TRUE)
    else()
      message(FATAL_ERROR "text-only test gate names an unregistered test: ${_ninfer_test}")
    endif()
  endforeach()
endif()

