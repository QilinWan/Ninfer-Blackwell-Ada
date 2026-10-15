# Two-device (tp 2) Op sources of ninfer_ops, included from ../basic_sources.cmake with one line.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../common/allreduce.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../common/peer_mailbox.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../launcher/argmax_split.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../launcher/concat_rows.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../launcher/topk_split.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../rmsnorm_rope/tp2/rmsnorm_rope_tp2.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../context_kv_materialize/tp2/context_kv_materialize_tp2.cpp"
)

# The two-device shards of upstream's fused projections: upstream's own sources compiled again
# with the shard's section output and entry points (see each directory's *_shard.h).
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/fp8_attn_input_shard_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/fp8_attn_input_shard_a16_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/fp8_attn_input_shard_a16_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/fp8_attn_input_shard_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/fp8_attn_input_shard_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/nvfp4_attn_input_shard_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/nvfp4_attn_input_shard_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/nvfp4_attn_input_shard_a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/nvfp4_attn_input_shard_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/nvfp4_attn_input_shard_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/fp8_gdn_input_shard_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/fp8_gdn_input_shard_matrix.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/fp8_gdn_input_shard_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/fp8_gdn_input_shard_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/nvfp4_gdn_input_shard_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/nvfp4_gdn_input_shard_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/nvfp4_gdn_input_shard_a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/nvfp4_gdn_input_shard_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/nvfp4_gdn_input_shard_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../linear_swiglu/tp2/fp8_linear_swiglu_half_a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../linear_swiglu/tp2/fp8_linear_swiglu_half_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../linear_swiglu/tp2/fp8_linear_swiglu_half_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../linear_swiglu/tp2/nvfp4_linear_swiglu_half_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../linear_swiglu/tp2/nvfp4_linear_swiglu_half_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../context_kv_materialize/tp2/materialize_shard_h4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/q8_dflash2_attn_input_shard_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/q8_dflash2_attn_input_head_block.cpp"
)
# Warp-specialized TMA kernels stay in the non-RDC archive, as upstream's (../CMakeLists.txt).
target_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/nvfp4_attn_input_shard_a4_tma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../gdn_input_proj/tp2/nvfp4_gdn_input_shard_a4_tma.cu"
)
