// sm_120a (Blackwell) replacement for the Ada-only E8 KV route.
//
// The rk4v4-e8 storage mode (E8 Conway-Sloane lattice keys, i4x16 values) exists because Ada
// has no native 4-bit KV format. Blackwell ships NVFP4, so this build drops the E8 route
// translation units (cmake/Sm120Compat.cmake) and answers the entry points the public op
// dispatches to with an explicit status, instead of compiling kernels for absent hardware:
// select the native NVFP4 KV cache (nvfp4 or k8v4) or INT8/FP8 KV on sm_120a.
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/e8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/e8/plan.h"

#include <cstddef>
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

std::size_t e8_kv_workspace_bytes(int, int, int, int, CausalAttentionExecutionEnvelope, int) {
    sm120a_route_unsupported("rk4v4-e8 causal attention");
}

void e8_kv_append_attention(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                            CausalAttentionExecutionEnvelope, WorkspaceArena&, Tensor&,
                            DeviceExecutionView) {
    sm120a_route_unsupported("rk4v4-e8 causal attention");
}

void e8_kv_cached_attention(const Tensor&, const Tensor&, float, const PagedKVLayerView&,
                            CausalAttentionExecutionEnvelope, WorkspaceArena&, Tensor&,
                            DeviceExecutionView) {
    sm120a_route_unsupported("rk4v4-e8 causal attention");
}

} // namespace ninfer::ops::detail
