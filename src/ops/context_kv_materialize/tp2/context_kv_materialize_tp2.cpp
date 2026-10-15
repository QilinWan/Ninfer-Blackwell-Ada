// Implements: include/ninfer/ops/tp2/context_kv_materialize.h. Our file.
// The four-KV-head block of context_kv_materialize: upstream's checks with the block's shapes,
// upstream's routes (ops/context_kv_materialize/launch.h) and upstream's kernels compiled for four
// heads (materialize_shard_h4.cu).
#include "core/weight.h"
#include "ninfer/ops/tp2/context_kv_materialize.h"

#include "core/layout.h"
#include "ops/context_kv_materialize/launch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace detail {
void context_kv_materialize_h4_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, ContextKVMaterializeRoute route,
    const Tensor& key_scratch, cudaStream_t stream);
} // namespace detail

namespace {

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kKVHeads    = kContextKVMaterializeBlockKVHeads;
constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kKVSize     = kKVHeads * kHeadDim;
constexpr std::int32_t kCapacity   = 2048;
constexpr std::int32_t kBlockWidth = 16;
constexpr const char* kOp          = "context_kv_materialize_head_block";

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, std::int32_t n3, std::size_t alignment, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != n3 || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_weight(const Weight& weight, const char* name) {
    constexpr std::uint64_t kCodeBytes =
        static_cast<std::uint64_t>(kKVSize) * static_cast<std::uint64_t>(kHidden);
    constexpr std::uint64_t kScaleBytes =
        static_cast<std::uint64_t>(kKVSize) * static_cast<std::uint64_t>(kHidden / 32) * 2U;
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != kKVSize || weight.k != kHidden ||
        weight.shape[0] != kKVSize || weight.shape[1] != kHidden ||
        weight.padded_shape[0] != kKVSize || weight.padded_shape[1] != kHidden ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < kCodeBytes + kScaleBytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_profile(std::int32_t width, std::int32_t batch) {
    if (batch < 1 || batch > 8 || width < 1) {
        throw std::invalid_argument(std::string(kOp) + ": invalid W/B");
    }
    if (width <= kBlockWidth) return;
    if (batch == 1 && width <= kCapacity) return;
    throw std::invalid_argument(std::string(kOp) + ": unsupported W/B profile");
}

void require_interval(std::int32_t batch, std::int32_t min_width, std::int32_t max_width) {
    if (batch < 1 || batch > 8 || min_width < 1 || max_width < min_width ||
        (batch == 1 && max_width > kCapacity) || (batch > 1 && max_width > kBlockWidth)) {
        throw std::invalid_argument(std::string(kOp) + " workspace: invalid interval");
    }
}

template <class Allocator>
Tensor allocate_key_scratch(Allocator& allocator, std::int32_t columns) {
    return allocator.alloc(
        DType::FP32, {kKVSize, columns, static_cast<std::int32_t>(kContextKVMaterializeLayers)});
}

void validate_cache(const CyclicKVCacheLayerView& cache, std::int32_t padded,
                    std::int32_t lane_capacity) {
    if (cache.capacity != kCapacity ||
        cache.padded_capacity != static_cast<std::uint32_t>(padded) ||
        cache.num_kv_heads != kKVHeads || cache.head_dim != kHeadDim ||
        cache.lane_capacity != lane_capacity || padded < kCapacity || lane_capacity <= 0) {
        throw std::invalid_argument(std::string(kOp) + ": invalid cyclic cache geometry");
    }
    require_tensor(cache.k, DType::BF16, kHeadDim, padded, kKVHeads, lane_capacity, 16, "cache K");
    require_tensor(cache.v, DType::FP16, kHeadDim, padded, kKVHeads, lane_capacity, 16, "cache V");
}

} // namespace

std::size_t context_kv_materialize_head_block_workspace_capacity_bytes(std::int32_t batch_size,
                                                                       std::int32_t min_width,
                                                                       std::int32_t max_width) {
    require_interval(batch_size, min_width, max_width);
    std::size_t capacity = 0;
    for (int count = 1; count <= max_width; ++count) {
        const int columns = count * batch_size;
        if (detail::context_kv_materialize_uses_scratch(
                detail::context_kv_materialize_route(columns))) {
            WorkspaceLayoutBuilder layout;
            (void)allocate_key_scratch(layout, columns);
            capacity = std::max(capacity, layout.peak_bytes(1));
        }
    }
    return capacity;
}

void context_kv_materialize_head_block(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace,
    cudaStream_t stream) {
    const std::int32_t width = context.ne[1];
    const std::int32_t batch = context.ne[2];
    require_profile(width, batch);
    require_tensor(context, DType::BF16, kHidden, width, batch, 1, 16, "context");
    require_tensor(positions, DType::I32, width, batch, 1, 1, alignof(std::int32_t), "positions");
    require_tensor(counts, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "counts");
    require_tensor(state_slots, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "state slots");
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(width) ||
        envelope.max_count > static_cast<std::uint32_t>(kCapacity)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid execution envelope");
    }
    if (layers.front().cache.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(kOp) + ": padded capacity exceeds int32");
    }
    const std::int32_t padded = static_cast<std::int32_t>(layers.front().cache.padded_capacity);
    const std::int32_t lane_capacity = layers.front().cache.lane_capacity;
    for (const ContextKVMaterializeLayerView& layer : layers) {
        require_weight(layer.key_weight, "key weight");
        require_weight(layer.value_weight, "value weight");
        require_tensor(layer.key_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, 4, "key norm weight");
        validate_cache(layer.cache, padded, lane_capacity);
    }
    if (envelope.max_count == 0) return;
    const int columns = envelope.max_count * batch;
    const auto route  = detail::context_kv_materialize_route(columns);
    auto scope        = workspace.scope();
    Tensor key_scratch;
    if (detail::context_kv_materialize_uses_scratch(route))
        key_scratch = allocate_key_scratch(workspace, columns);
    detail::context_kv_materialize_h4_launch(context, positions, counts, state_slots, layers,
                                             envelope, route, key_scratch, stream);
}

} // namespace ninfer::ops
