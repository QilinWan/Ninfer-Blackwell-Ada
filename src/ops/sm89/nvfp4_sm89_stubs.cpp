// Ada (sm_89) replacement for the Blackwell-only NVFP4 and K8V4 kernel routes.
//
// Those routes need FP4 code conversion (`cvt.rn.satfinite.e2m1x2.f32`), block-scaled MMA and TMA
// with `setmaxnreg`, none of which exist on Ada, so cmake/Sm89Compat.cmake drops their translation
// units. The dispatchers still name their launchers, and every reference here answers with the
// missing capability instead of silently computing something wrong.
//
// Supported on sm_89: groupwise INT (Q4/Q5/Q6/Q8) weights, FP8 A8/A16, BF16, INT8 and FP8 KV.
// Not supported: NVFP4 weights, NVFP4 and K8V4 KV cache - both need a Blackwell build.
#include <cuda_bf16.h>

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void sm89_route_unsupported(const char* route) {
    throw std::invalid_argument(
        std::string("NInfer sm_89 build: the ") + route +
        " route needs Blackwell (sm_120a) instructions. Run a groupwise-int (.ninfer) artifact on "
        "Ada and select an INT8 or FP8 KV cache; NVFP4 weights and NVFP4/K8V4 KV need the sm_120a "
        "build");
}

void nvfp4_a16_unsupported(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    sm89_route_unsupported("nvfp4 a16 linear");
}

void nvfp4_a4_unsupported(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace, cudaStream_t) {
    sm89_route_unsupported("nvfp4 w4a4 linear");
}

// No token interval prefers the FP4 W4A4 route on Ada, so planning never selects it.
[[nodiscard]] bool nvfp4_never_uses_a4(std::int32_t, std::int32_t) { return false; }

} // namespace

// The shape metadata itself is architecture-neutral: workspace sizing and shape resolution keep
// working, only the kernel entry points are replaced.
const Nvfp4LinearShape kNvfp4N14336K5120{14336, 5120, &nvfp4_a16_unsupported, &nvfp4_a4_unsupported,
                                         &nvfp4_never_uses_a4};
const Nvfp4LinearShape kNvfp4N16384K5120{16384, 5120, &nvfp4_a16_unsupported, &nvfp4_a4_unsupported,
                                         &nvfp4_never_uses_a4};
const Nvfp4LinearShape kNvfp4N34816K5120{34816, 5120, &nvfp4_a16_unsupported, &nvfp4_a4_unsupported,
                                         &nvfp4_never_uses_a4};
const Nvfp4LinearShape kNvfp4N5120K6144{5120, 6144, &nvfp4_a16_unsupported, &nvfp4_a4_unsupported,
                                        &nvfp4_never_uses_a4};
const Nvfp4LinearShape kNvfp4N5120K17408{5120, 17408, &nvfp4_a16_unsupported, &nvfp4_a4_unsupported,
                                         &nvfp4_never_uses_a4};

void launch_nvfp4_w4a4_quantize(const Tensor&, const Weight&, Nvfp4W4a4Workspace, Nvfp4ScaleLayout,
                                cudaStream_t) {
    sm89_route_unsupported("launch_nvfp4_w4a4_quantize");
}

void launch_nvfp4_w4a4_tma_linear(Nvfp4GeometryId, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    sm89_route_unsupported("launch_nvfp4_w4a4_tma_linear");
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, std::int32_t, float,
                                     cudaStream_t) {
    sm89_route_unsupported("launch_nvfp4_w4a4_tma_attention");
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    sm89_route_unsupported("launch_nvfp4_w4a4_tma_gdn");
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4GeometryId, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    sm89_route_unsupported("launch_nvfp4_w4a4_tma_linear_add");
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                         std::int32_t, float, cudaStream_t) {
    sm89_route_unsupported("launch_nvfp4_linear_swiglu_w4a4_tma");
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                     cudaStream_t) {
    sm89_route_unsupported("nvfp4_linear_swiglu_w4a4_launch");
}

void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Nvfp4W4a4Workspace,
                                 cudaStream_t) {
    sm89_route_unsupported("nvfp4_gdn_input_w4a4_launch");
}

void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    sm89_route_unsupported("nvfp4_attn_input_w4a4_launch");
}

void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    sm89_route_unsupported("nvfp4_linear_add_w4a4_launch");
}

void causal_attention_small_t_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                           const Tensor&, const Tensor&, const Tensor&, float,
                                           PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                           std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                           Tensor&, cudaStream_t) {
    sm89_route_unsupported("causal_attention_small_t_nvfp4_launch");
}

void causal_attention_small_t_k8v4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                          std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                          Tensor&, cudaStream_t) {
    sm89_route_unsupported("causal_attention_small_t_k8v4_launch");
}

void causal_attention_prompt_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, Tensor&, cudaStream_t) {
    sm89_route_unsupported("causal_attention_prompt_nvfp4_launch");
}

void causal_attention_prompt_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                         Tensor&, cudaStream_t) {
    sm89_route_unsupported("causal_attention_prompt_k8v4_launch");
}

void causal_attention_prompt_nvfp4_attention_launch(const Tensor&, const Tensor&, float,
                                                    const PagedKVLayerView&, Tensor&,
                                                    cudaStream_t) {
    sm89_route_unsupported("causal_attention_prompt_nvfp4_attention_launch");
}

void causal_attention_prompt_k8v4_attention_launch(const Tensor&, const Tensor&, float,
                                                   const PagedKVLayerView&, Tensor&, cudaStream_t) {
    sm89_route_unsupported("causal_attention_prompt_k8v4_attention_launch");
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&,
                                                  CausalAttentionExecutionEnvelope, Tensor&,
                                                  Tensor&, Tensor&, Tensor&, cudaStream_t) {
    sm89_route_unsupported("causal_attention_cached_small_t_nvfp4_launch");
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&,
                                                 CausalAttentionExecutionEnvelope, Tensor&, Tensor&,
                                                 Tensor&, Tensor&, cudaStream_t) {
    sm89_route_unsupported("causal_attention_cached_small_t_k8v4_launch");
}

void kv_cache_append_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                  cudaStream_t) {
    sm89_route_unsupported("kv_cache_append_nvfp4_launch");
}

void kv_cache_append_nvfp4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    sm89_route_unsupported("kv_cache_append_nvfp4_batch_launch");
}

void kv_cache_append_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                 cudaStream_t) {
    sm89_route_unsupported("kv_cache_append_k8v4_launch");
}

void kv_cache_append_k8v4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    sm89_route_unsupported("kv_cache_append_k8v4_batch_launch");
}
} // namespace ninfer::ops::detail
