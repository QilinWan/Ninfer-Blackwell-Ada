find_package(CUDAToolkit REQUIRED)

# The Blackwell target keeps the upstream static staging unions. Outside the tiled wide-column
# schedules the Q8 k-split and NVFP4 routes declare a >48 KB union with a plain `__shared__`
# (src/ops/linear/q8/q8_ksplit_mma.cuh). CUDA 12.x ptxas/nvlink still applies the 48 KB static
# shared-memory limit to sm_120a, so every translation unit compiles and the Ops device link then
# fails with "uses too much shared data (... 0xc000 max)". CUDA 13.0 raised that limit; 12.8 and
# 12.9 both still fail. Only the Ada branch needs the 12.8 floor.
if(NOT NINFER_SM89_BUILD AND CUDAToolkit_VERSION VERSION_LESS "13.0")
  message(FATAL_ERROR
    "The sm_120a build needs CUDA 13.0 or newer (found ${CUDAToolkit_VERSION}). CUDA 12.x "
    "ptxas/nvlink caps static shared memory at 48 KB for sm_120a, so the ninfer_ops device link "
    "fails with 'uses too much shared data (0xc000 max)'. CUDA 13.1 or 13.2 is the validated "
    "toolkit.")
endif()

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
if(NINFER_ENABLE_VISION)
  # The media decoder uses the FFmpeg 6 packet side-data API (av_packet_side_data_get). Ubuntu
  # 22.04 and Debian 11 ship FFmpeg 4/5, where decode.cpp fails to compile; require the floor at
  # configure time and point at the option instead.
  pkg_check_modules(FFMPEG IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  if(NOT FFMPEG_FOUND)
    if(FFMPEG_libavcodec_VERSION)
      set(NINFER_FFMPEG_FOUND "found libavcodec ${FFMPEG_libavcodec_VERSION}")
    else()
      set(NINFER_FFMPEG_FOUND "no libavcodec >= 60 was reported by pkg-config")
    endif()
    message(FATAL_ERROR
      "NINFER_ENABLE_VISION=ON needs FFmpeg 6 or newer development packages (${NINFER_FFMPEG_FOUND})."
      " Install libavformat-dev/libavcodec-dev/libswscale-dev from Ubuntu 24.04 or newer, or"
      " configure with -DNINFER_ENABLE_VISION=OFF for a text-only engine.")
  endif()
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
