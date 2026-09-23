if(NINFER_ENABLE_VISION)
  add_library(ninfer_media_decode STATIC
    decode/decode.cpp
  )
  target_link_libraries(ninfer_media_decode PRIVATE PkgConfig::FFMPEG)
else()
  add_library(ninfer_media_decode STATIC
    decode/stub.cpp
  )
endif()
ninfer_internal_includes(ninfer_media_decode)
