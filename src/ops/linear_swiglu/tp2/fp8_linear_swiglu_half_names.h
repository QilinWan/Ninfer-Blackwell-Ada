// Renames that turn upstream's FP8 linear_swiglu sources into the [17408,5120] half's when a
// translation unit of this directory includes them. Include it once, after the family's headers
// and right before the upstream source. The A8 launcher and its partial reservation read the rows
// from the weight and are shared, not renamed.

#define Fp8N34816K5120 Fp8N17408K5120
#define fp8_linear_swiglu_decode_launch fp8_linear_swiglu_half_decode_launch
#define fp8_linear_swiglu_small_t_launch fp8_linear_swiglu_half_small_t_launch
#define fp8_linear_swiglu_matrix_launch fp8_linear_swiglu_half_matrix_launch
#define fp8_linear_swiglu_workspace_capacity_bytes fp8_linear_swiglu_half_workspace_capacity_bytes
#define fp8_linear_swiglu_dispatch fp8_linear_swiglu_half_dispatch
