// Renames that turn upstream's NVFP4 gdn_input_proj sources into the [8192,5120] shard's when a
// translation unit of this directory includes them. Include it once, after
// nvfp4_gdn_input_shard_prelude.h and right before the upstream source.

#define Nvfp4GdnInputOutput Nvfp4GdnInputShardOutput
#define nvfp4_gdn_input_a16_launch nvfp4_gdn_input_shard_a16_launch
#define nvfp4_gdn_input_decode_launch nvfp4_gdn_input_shard_decode_launch
#define nvfp4_gdn_input_small_t_launch nvfp4_gdn_input_shard_small_t_launch
#define nvfp4_gdn_input_a4_launch nvfp4_gdn_input_shard_a4_launch
#define launch_nvfp4_a4_tma_gdn launch_nvfp4_a4_tma_gdn_shard
#define nvfp4_gdn_input_workspace_capacity_bytes nvfp4_gdn_input_shard_workspace_capacity_bytes
#define nvfp4_gdn_input_dispatch nvfp4_gdn_input_shard_dispatch
