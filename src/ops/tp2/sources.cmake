# Two-device (tp 2) Op sources of ninfer_ops.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../softmax_attention/sliding_window/tp2/launch_tp2.inc"
  "${CMAKE_CURRENT_LIST_DIR}/../rmsnorm_rope/tp2/rmsnorm_rope_tp2.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/q8_dflash2_attn_input_shard_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../attn_input_proj/tp2/q8_dflash2_attn_input_head_block.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../context_kv_materialize/tp2/context_kv_materialize_tp2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../context_kv_materialize/tp2/materialize_shard_h4.cu"
)
