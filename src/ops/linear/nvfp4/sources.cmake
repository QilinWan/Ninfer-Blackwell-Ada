target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1024_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n8192_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n24576_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k12288.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k8192.cu"
)

target_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4_tma.cu"
)
