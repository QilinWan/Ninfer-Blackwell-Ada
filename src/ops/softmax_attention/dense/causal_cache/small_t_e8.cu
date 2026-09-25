// ninfer::ops::detail - E8 Conway-Sloane (rk4v4-e8) split-KV small-T launch ownership.
// Ada-only route: the sm_120a build drops this translation unit and links an explicit
// unsupported stub (cmake/Sm120Compat.cmake).
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_e8.cuh"
#include "ops/softmax_attention/dense/causal_cache/e8_output_rotate.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_e8_partial(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                       PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                       std::int32_t logical_capacity, std::int32_t implementation_window,
                       std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                       Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kCausalHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                causal_attention_small_t_e8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                         MinBlocksPerSm, KeyBlock, DynamicArena,
                                                         MultiBatch, Masked, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        causal_attention_small_t_e8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                                KeyBlock, DynamicArena, MultiBatch, Masked,
                                                CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data),
                static_cast<std::uint8_t*>(cache_k.data), static_cast<std::uint8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    // The E8 arena (K codes, V codes, K i8, V FP16) is byte-identical in size to the int8
    // arena, so the tier table is the int8 one.
    if constexpr (TokenTile >= 6) {
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, bool MultiBatch, bool Masked>
void launch_e8_reduce(const Tensor& pos, const CausalSmallTInvocation& invocation,
                      std::int32_t splits, const Tensor& partial_acc, const Tensor& partial_m,
                      const Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    constexpr int kBlock  = 256;
    constexpr int kDChunk = Geometry::QHeads == 24 ? 256 : 64;
    const dim3 grid(Geometry::QHeads, div_up(kCausalHeadDim, kDChunk),
                    invocation.width * invocation.batch_size);
    const auto launch = [&]<bool Offset>() {
        causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, false, MultiBatch, Masked,
                                                      Offset>
            <<<grid, kBlock, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns
                    ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                    : nullptr,
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
    };
    if (invocation.column_begin == 0) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_e8_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                            float scale, PagedKVBatchLayerView cache,
                                            const CausalSmallTInvocation& invocation,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits                = causal_attention_split_capacity(
        Geometry::QHeads, invocation.width, cache.storage, envelope, invocation.batch_size);

    const auto launch_partial = [&]<int Tokens, bool MultiBatch, bool Masked>() {
        launch_e8_partial<Geometry, Tokens, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
    };
    const auto dispatch_metadata = [&]<int Tokens>() {
        const bool masked = invocation.valid_columns != nullptr;
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_partial.template operator()<Tokens, false, true>();
            } else {
                launch_partial.template operator()<Tokens, false, false>();
            }
        } else if (masked) {
            launch_partial.template operator()<Tokens, true, true>();
        } else {
            launch_partial.template operator()<Tokens, true, false>();
        }
    };

    switch (invocation.width) {
    case 1:
        dispatch_metadata.template operator()<1>();
        break;
    case 2:
        dispatch_metadata.template operator()<2>();
        break;
    case 3:
        dispatch_metadata.template operator()<3>();
        break;
    case 4:
        dispatch_metadata.template operator()<4>();
        break;
    case 5:
        dispatch_metadata.template operator()<5>();
        break;
    case 6:
        dispatch_metadata.template operator()<6>();
        break;
    case 7:
        if constexpr (Geometry::QHeads == 24) {
            dispatch_metadata.template operator()<7>();
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    case 8:
        if constexpr (Geometry::QHeads == 24) {
            dispatch_metadata.template operator()<8>();
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    default:
        throw std::invalid_argument("causal_attention_small_t_e8_launch: unsupported T");
    }

    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_e8_reduce<Geometry, false, true>(pos, invocation, splits, partial_acc,
                                                    partial_m, partial_l, out, stream);
        } else {
            launch_e8_reduce<Geometry, false, false>(pos, invocation, splits, partial_acc,
                                                     partial_m, partial_l, out, stream);
        }
    } else if (masked) {
        launch_e8_reduce<Geometry, true, true>(pos, invocation, splits, partial_acc, partial_m,
                                               partial_l, out, stream);
    } else {
        launch_e8_reduce<Geometry, true, false>(pos, invocation, splits, partial_acc, partial_m,
                                                partial_l, out, stream);
    }

    // The E8 route stores V in the rotated domain (see e8_output_rotate.cuh), so the reduce
    // output has to be un-rotated before the caller sees it.
    e8_inverse_rotate_output_launch<Geometry::QHeads, kCausalHeadDim>(
        out, invocation.width, invocation.full_width, invocation.column_begin,
        invocation.valid_columns, invocation.batch_size, stream);
}

} // namespace

void causal_attention_small_t_e8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_e8_launch_for<CausalD256H24Kv4>(
            q, input, pos, scale, cache, invocation, envelope, partial_acc, partial_m, partial_l,
            out, stream);
        return;
    }
    causal_attention_small_t_e8_launch_for<CausalD256H16Kv2>(
        q, input, pos, scale, cache, invocation, envelope, partial_acc, partial_m, partial_l, out,
        stream);
}

void causal_attention_cached_small_t_e8_launch(const Tensor& q, const Tensor& pos, float scale,
                                               const PagedKVLayerView& cache,
                                               CausalAttentionExecutionEnvelope envelope,
                                               Tensor& partial_acc, Tensor& partial_m,
                                               Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_e8_launch_for<CausalD256H24Kv4>(
            q, input, pos, scale, batch_cache, invocation, envelope, partial_acc, partial_m,
            partial_l, out, stream);
        return;
    }
    causal_attention_small_t_e8_launch_for<CausalD256H16Kv2>(
        q, input, pos, scale, batch_cache, invocation, envelope, partial_acc, partial_m, partial_l,
        out, stream);
}

} // namespace ninfer::ops::detail
