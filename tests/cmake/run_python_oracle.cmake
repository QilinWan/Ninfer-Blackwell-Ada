# run_python_oracle.cmake — run one python oracle and require BOTH facts:
#   the process exits 0, AND its stdout contains the [PASS] marker.
# ctest alone cannot state that: with PASS_REGULAR_EXPRESSION the return code is ignored, and
# without it a printed failure is invisible. Register python oracles through this runner.
# Usage: ${CMAKE_COMMAND} -DPYTHON=<exe> -DORACLE=<script> [-DARGS=a;b] -P run_python_oracle.cmake
if(NOT DEFINED PYTHON OR NOT DEFINED ORACLE)
  message(FATAL_ERROR "run_python_oracle.cmake needs -DPYTHON=<exe> -DORACLE=<script>")
endif()
execute_process(COMMAND ${PYTHON} -B ${ORACLE} ${ARGS}
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message(STATUS "${out}${err}")
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "oracle exited rc=${rc}: ${ORACLE}")
endif()
if(NOT out MATCHES "\\[PASS\\]")
  message(FATAL_ERROR "oracle printed no [PASS] marker: ${ORACLE}")
endif()
message(STATUS "oracle [PASS]: ${ORACLE}")
