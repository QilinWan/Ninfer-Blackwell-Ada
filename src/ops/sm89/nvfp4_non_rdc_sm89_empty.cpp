// The non-RDC archive carries SM120 warp-specialized kernels (setmaxnreg + TMA).
// Ada builds keep the target alive so the family manifests stay unchanged, but it
// contributes no device code here.
namespace ninfer::ops::sm89 {
inline constexpr bool kNvfp4NonRdcOmitted = true;
} // namespace ninfer::ops::sm89
