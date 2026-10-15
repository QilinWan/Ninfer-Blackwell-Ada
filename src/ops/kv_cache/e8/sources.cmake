# E8 Conway-Sloane lattice KV codec (P1 landing: header-only, device arithmetic; the rk4v4-e8 /
# rk2v4-e8 storage dispatch and attention-kernel grafts land in P2).
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/e8_lattice.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/e8_root_codec.cuh"
)
