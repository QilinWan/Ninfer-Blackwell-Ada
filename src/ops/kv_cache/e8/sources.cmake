# E8 Conway-Sloane lattice KV codec: lattice projection (rk4v4-e8), cylinder root/rad-axis codes
# (rk2v4-e8, storage dispatch lands later), and the shared i4 code helpers.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/e8_lattice.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/e8_root_codec.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/e8_i4_code.cuh"
)
