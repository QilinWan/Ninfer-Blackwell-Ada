# Two-device (tp == 2) execution sources of ninfer_model_runtime, included from
# ../../execution_sources.cmake with one line.
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/attention_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/draft_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ffn_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gdn_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mtp_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../tp.cpp"
)
