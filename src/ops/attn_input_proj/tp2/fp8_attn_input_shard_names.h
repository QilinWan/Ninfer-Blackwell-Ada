// Renames that turn upstream's FP8 attn_input_proj sources into the [7168,5120] shard's when a
// translation unit of this directory includes them. Not a header in the usual sense: include it
// once, after the family's own headers (fp8_attn_input_shard_prelude.h) and right before the
// upstream source, so that only that source's code sees the substitutions.

#define Fp8AttentionInputOutput Fp8AttnInputShardOutput
#define fp8_attn_input_a16_small_mma_launch fp8_attn_input_shard_a16_small_mma_launch
#define fp8_attn_input_a16_gemm_launch fp8_attn_input_shard_a16_gemm_launch
#define fp8_attn_input_decode_launch fp8_attn_input_shard_decode_launch
#define fp8_attn_input_a8_launch fp8_attn_input_shard_a8_launch
#define fp8_attn_input_partial_capacity_bytes fp8_attn_input_shard_partial_capacity_bytes
#define fp8_attn_input_workspace_capacity_bytes fp8_attn_input_shard_workspace_capacity_bytes
#define fp8_attn_input_dispatch fp8_attn_input_shard_dispatch
