// Renames that turn upstream's q8/q8_dflash2_attn_input.cu into the [3072,5120] shard's when a
// translation unit of this directory includes it. Include it once, after the family's headers
// (q8_dflash2_attn_input_shard_prelude.h) and right before the upstream source, so that only that
// source's code sees the substitutions. The source's Geometry only supplies K (5120, unchanged);
// the rows come from the weight.

#define LinearBf16SegmentedOutput Q8DFlash2AttnInputShardOutput
#define q8_dflash2_attn_input_small_t_launch q8_dflash2_attn_input_shard_small_t_launch
#define q8_dflash2_attn_input_mma_r32_c64_launch q8_dflash2_attn_input_shard_mma_r32_c64_launch
#define q8_dflash2_attn_input_mma_r64_c128_launch q8_dflash2_attn_input_shard_mma_r64_c128_launch
#define q8_dflash2_attn_input_mma_r16_c64_k128_launch q8_dflash2_attn_input_shard_mma_r16_c64_k128_launch
#define q8_dflash2_attn_input_mma_r32_c32_k128_launch q8_dflash2_attn_input_shard_mma_r32_c32_k128_launch
#define q8_dflash2_attn_input_mma_r32_c64_k128_launch q8_dflash2_attn_input_shard_mma_r32_c64_k128_launch
