# run_python_oracle.cmake — run one python oracle and require BOTH facts:
#   the process exits 0, AND its stdout contains the [PASS] marker.
# ctest alone cannot state that: with PASS_REGULAR_EXPRESSION the return code is ignored, and
# without it a printed failure is invisible. Register python oracles through this runner.
# Usage: ${CMAKE_COMMAND} -DPYTHON=<exe> -DORACLE=<script> [-DARGS=a;b] -P run_python_oracle.cmake
if(NOT DEFINED ORACLE)
  message(FATAL_ERROR "run_python_oracle.cmake needs -DORACLE=<script>")
endif()
# An oracle with a numerical dependency cannot run under every interpreter CMake may have found:
# this machine's configured interpreter is a python3.14 without numpy, while /usr/bin/python3 3.10
# carries it.  The candidates are listed here (not passed on the command line: ctest splits a
# -D<list> argument into separate argv entries, and quotes a -D<value with spaces> into the value)
# and the first interpreter that satisfies the dependency runs the oracle.  When none does, the
# test fails loudly: a skip would hide the mathematical face behind an unmeasured green.
if(NOT DEFINED IMPORT_CHECK)
  set(IMPORT_CHECK "import numpy")
endif()
if(NOT DEFINED PYTHON)
  set(ninfer_oracle_candidates ${Python3_EXECUTABLE} /usr/bin/python3 python3 python3.12 python3.11)
  foreach(cand ${ninfer_oracle_candidates})
    if(NOT cand)
      continue()
    endif()
    execute_process(COMMAND ${cand} -c "${IMPORT_CHECK}" RESULT_VARIABLE dep_rc
                    OUTPUT_QUIET ERROR_QUIET)
    if(dep_rc EQUAL 0)
      set(PYTHON ${cand})
      break()
    endif()
  endforeach()
endif()
if(NOT DEFINED PYTHON)
  message(FATAL_ERROR "no interpreter satisfies the oracle dependency: ${IMPORT_CHECK}; candidates: ${ninfer_oracle_candidates}")
endif()
message(STATUS "oracle interpreter: ${PYTHON}")
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
