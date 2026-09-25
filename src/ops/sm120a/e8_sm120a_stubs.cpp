// sm_120a (Blackwell) replacement for the Ada-only E8 KV route.
//
// The rk4v4-e8 storage mode (E8 Conway-Sloane lattice keys, i4x16 values) exists because Ada
// has no native 4-bit KV format. Blackwell ships NVFP4, so this build drops the E8 kernel
// translation units and answers the route with an explicit status instead of compiling dead
// kernels; select the native Nvfp4Group16 or Fp8KeyNvfp4Value KV cache on sm_120a.
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void sm120a_route_unsupported(const char* route) {
    throw std::invalid_argument(
        std::string("NInfer sm_120a build: the ") + route +
        " route is an Ada (sm_89) capability. On Blackwell select the native NVFP4 KV cache "
        "(nvfp4 or k8v4) or INT8/FP8 KV");
}

} // namespace

void kv_cache_append_e8_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                               cudaStream_t) {
    sm120a_route_unsupported("rk4v4-e8 KV append");
}

void kv_cache_append_e8_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                     const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    sm120a_route_unsupported("rk4v4-e8 KV batch append");
}

void causal_attention_small_t_e8_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, float,
                                        PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                        std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                        Tensor&, cudaStream_t) {
    sm120a_route_unsupported("rk4v4-e8 small-T attention");
}

void causal_attention_cached_small_t_e8_launch(const Tensor&, const Tensor&, float,
                                               const PagedKVLayerView&,
                                               CausalAttentionExecutionEnvelope, Tensor&, Tensor&,
                                               Tensor&, Tensor&, cudaStream_t) {
    sm120a_route_unsupported("rk4v4-e8 cached small-T attention");
}

void causal_attention_prompt_e8_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                       Tensor&, cudaStream_t) {
    sm120a_route_unsupported("rk4v4-e8 prompt attention");
}

void causal_attention_prompt_e8_attention_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&, Tensor&, cudaStream_t) {
    sm120a_route_unsupported("rk4v4-e8 prompt attention");
}

} // namespace ninfer::ops::detail
