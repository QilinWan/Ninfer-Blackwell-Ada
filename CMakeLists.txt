cmake_minimum_required(VERSION 3.28)

# Upstream NInfer compiles exactly one architecture: sm_120a (Blackwell). This
# repository additionally targets Ada (sm_89: RTX 4090 / 4090D / 4080 SUPER /
# 4070 family). Set and validate the architecture before CUDA compiler detection
# so CMake never injects a generic host-toolchain default or accepts an
# unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architecture to build: 120a or 89")
endif()
set(NINFER_SM89_BUILD OFF)
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  set(NINFER_SM89_BUILD ON)
  add_compile_definitions(NINFER_SM89=1)
elseif(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=120a (upstream, Blackwell) or 89 (Ada); got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
# Media decoding (image/video input for the vision encoder) links FFmpeg. Downstream
# users who only run text generation can configure without the FFmpeg development
# packages; the vision route then fails with an explicit message instead of a
# configure-time dependency error.
option(NINFER_ENABLE_VISION "Build the FFmpeg-backed media decoder" ON)
# Links the engine archives into an executable on hosts without a GPU. Useful in CI and for
# downstream packagers: it catches unresolved kernel routes that static archives tolerate.
option(NINFER_BUILD_LINK_SMOKE "Build the engine link smoke executable" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(NINFER_BUILD_LINK_SMOKE)
  add_subdirectory(tools/link_smoke)
endif()
