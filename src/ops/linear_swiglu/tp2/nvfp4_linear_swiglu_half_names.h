// Renames that turn upstream's NVFP4 linear_swiglu sources into the [17408,5120] half's when a
// translation unit of this directory includes them. Include it once, after the family's headers
// and right before the upstream source. The decode, small-T and TMA launchers read the rows from the
// weight and are shared, not renamed (the small-T one carries this fork's 70-SM sliced-K schedules,
// 91582814, for both problems).

#define Nvfp4N34816K5120 Nvfp4N17408K5120
#define nvfp4_linear_swiglu_a4_launch nvfp4_linear_swiglu_half_a4_launch
#define nvfp4_linear_swiglu_workspace_capacity_bytes nvfp4_linear_swiglu_half_workspace_capacity_bytes
#define nvfp4_linear_swiglu_dispatch nvfp4_linear_swiglu_half_dispatch
