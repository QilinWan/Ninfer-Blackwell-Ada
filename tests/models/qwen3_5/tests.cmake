ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_native_transactions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_native_transactions.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_preemption_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_grammar_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_grammar_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_tools_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tools_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_runtime ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES ninfer_engine ninfer_core)

add_test(NAME ninfer_qwen3_5_agent_continuation_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_agent_continuation_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=agent-continuation")

# A real-model test owns the single GPU while its artifact is resident.
set(ninfer_qwen3_5_real_tests
  ninfer_qwen3_5_loading_real_test
  ninfer_qwen3_5_native_transactions_test
  ninfer_qwen3_5_prefix_real_test
  ninfer_qwen3_5_agent_continuation_real_test
  ninfer_qwen3_5_preemption_real_test
  ninfer_qwen3_5_grammar_real_test
  ninfer_qwen3_5_tools_real_test
  ninfer_qwen3_5_score_real_test
  ninfer_qwen3_5_vision_workspace_test
  ninfer_qwen3_5_dflash2_real_test
  ninfer_qwen3_5_dflash_prefill_real_test
  ninfer_qwen3_5_moe_real_test
  ninfer_qwen3_5_dflash_real_test)
set_tests_properties(${ninfer_qwen3_5_real_tests} PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real")

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  ninfer_qwen3_5_context_store_test
  ninfer_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

ninfer_add_test(ninfer_qwen3_5_tool_constraints_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_constraints.cpp"
  LIBRARIES ninfer_model_runtime ninfer_grammar ninfer::json)

add_test(NAME ninfer_qwen3_5_tool_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_TOOL_PROBE=$<TARGET_FILE:ninfer_qwen3_5_tool_constraints_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_tool_schema.py")

ninfer_add_test(ninfer_qwen3_5_shard_map_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_shard_map.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_sharded_load_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sharded_load_real.cpp"
  LIBRARIES ninfer_model_runtime)

add_test(NAME ninfer_qwen3_5_sharded_load_mtp_real_test
  COMMAND ninfer_qwen3_5_sharded_load_real_test mtp)

set_tests_properties(
  ninfer_qwen3_5_sharded_load_real_test
  ninfer_qwen3_5_sharded_load_mtp_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_text_context_tp2_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_text_context_tp2.cpp"
  LIBRARIES ninfer_model_runtime)

add_test(NAME ninfer_qwen3_5_text_context_tp2_real_test
  COMMAND ninfer_qwen3_5_text_context_tp2_test real)

set_tests_properties(
  ninfer_qwen3_5_text_context_tp2_test
  ninfer_qwen3_5_text_context_tp2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_engine_tp2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tp2_real.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_engine_tp2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_engine_mtp_tp2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_mtp_tp2_real.cpp"
  LIBRARIES ninfer_engine ninfer_core)

add_test(NAME ninfer_qwen3_5_engine_mtp_tp2_optimized_real_test
  COMMAND ninfer_qwen3_5_engine_mtp_tp2_real_test optimized)

set_tests_properties(
  ninfer_qwen3_5_engine_mtp_tp2_real_test
  ninfer_qwen3_5_engine_mtp_tp2_optimized_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_engine_dflash2_tp2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_tp2_real.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_engine_dflash2_tp2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Vision at tp 2: one Engine holds the model across both devices; the tower lives whole on
# --vision-device. Three legs share one binary: parity (text answer with and without Vision),
# MTP and DFlash2 speculative runs with Vision. Skips without a second device or an artifact.
ninfer_add_test(ninfer_qwen3_5_engine_vision_tp2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_vision_tp2_real.cpp"
  LIBRARIES ninfer_engine ninfer_core)

add_test(NAME ninfer_qwen3_5_engine_vision_tp2_mtp_real_test
  COMMAND ninfer_qwen3_5_engine_vision_tp2_real_test mtp)

add_test(NAME ninfer_qwen3_5_engine_vision_tp2_dflash2_real_test
  COMMAND ninfer_qwen3_5_engine_vision_tp2_real_test dflash2)

set_tests_properties(
  ninfer_qwen3_5_engine_vision_tp2_real_test
  ninfer_qwen3_5_engine_vision_tp2_mtp_real_test
  ninfer_qwen3_5_engine_vision_tp2_dflash2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)
