#pragma once

// E8 Conway-Sloane KV-cache i4 code helpers shared by standalone append and causal-attention
// decode: 4-bit two's-complement codes packed two per byte (D/2 bytes per row), plus one
// per-token FP16 scale per row (ks = row absmax / 7, so a code of +/-7 covers the whole row).
// The code planes reuse paged element addressing with a head_dim/2 leading extent; the scale
// plane keeps the same per-token layout as the FP8 row codec.
//
// Code-domain contract (upstream sergiuszm small_t_i8.cuh fill path, ported verbatim):
//   * K: rotated row scaled by 1/ks, E8-projected per 8-dim subspace, rint() into the code,
//     clamped to [-8, 7]; the D8+0.5 half-coset is deliberately collapsed by the rint - it is
//     the mode's documented storage semantics, not an extra approximation.
//   * V: rotated row scaled by 1/ks, plain rint() into the code, clamped to [-7, 7].
//   * decode: value = code * ks.

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/e8/e8_lattice.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kE8I4HeadDim     = 256;
inline constexpr int kE8I4CodeExtent  = kE8I4HeadDim / 2;
inline constexpr int kE8I4ScaleExtent = 1;
inline constexpr float kE8I4ScaleRange = 7.0F;

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_e8_i4_code_index(int physical_page, int kv_head,
                                                                  int packed_d, int page_offset) {
    return paged_kv_element_offset<kE8I4CodeExtent, Geometry::KVHeads>(physical_page, kv_head,
                                                                       page_offset, packed_d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_e8_i4_scale_index(int physical_page, int kv_head,
                                                                   int page_offset) {
    return paged_kv_element_offset<kE8I4ScaleExtent, Geometry::KVHeads>(physical_page, kv_head,
                                                                        page_offset, 0);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_e8_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kE8I4HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheE8ScaleParams {
    __half scale;
    float inverse_scale;
};

// Per-token row scale: ks = absmax / 7 in FP16, bounded to the FP16 normal range the way the
// FP8 row codec bounds its scale, so a pathological row saturates its codes instead of
// producing an infinite scale that zeroes the whole row.
__device__ __forceinline__ KVCacheE8ScaleParams kv_cache_e8_scale_params(float absmax) {
    if (absmax == 0.0F) { return {.scale = __float2half_rn(0.0F), .inverse_scale = 0.0F}; }
    const float raw_scale    = absmax / kE8I4ScaleRange;
    const float bounded      = fminf(65504.0F, fmaxf(0x1p-24F, raw_scale));
    const __half scale       = __float2half_rn(bounded);
    const float represented  = __half2float(scale);
    return {.scale = scale, .inverse_scale = represented > 0.0F ? 1.0F / represented : 0.0F};
}

__device__ __forceinline__ std::int8_t kv_cache_e8_i4_quant_code(float x, float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    const int q = __float2int_rn(x * inverse_scale);
    return static_cast<std::int8_t>((q < -7) ? -7 : ((q > 7) ? 7 : q));
}

// E8 lattice key code: the projected lattice point, already in the absmax/7 code domain,
// rounded to the nearest integer (the documented half-coset collapse); clamped to [-8, 7] so a
// point at +7.5 still lands inside the 4-bit range.
__device__ __forceinline__ std::int8_t kv_cache_e8_lattice_code(float scaled_projected) {
    const int q = static_cast<int>(rintf(scaled_projected));
    return static_cast<std::int8_t>((q < -8) ? -8 : ((q > 7) ? 7 : q));
}

__device__ __forceinline__ std::uint8_t kv_cache_e8_pack_i4(std::int8_t lo, std::int8_t hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0Fu) |
                                     ((static_cast<unsigned>(hi) & 0x0Fu) << 4));
}

__device__ __forceinline__ std::int8_t kv_cache_e8_unpack_i4(std::uint8_t packed, int high) {
    const unsigned nibble = high ? (packed >> 4) : (packed & 0x0Fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

// 16 codes from 8 packed bytes (a 128-byte row decodes with 8 of these).
__device__ __forceinline__ void kv_cache_e8_unpack_i4x16(const std::uint8_t* src8,
                                                         std::int8_t* dst16) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst16[2 * i]     = kv_cache_e8_unpack_i4(src8[i], 0);
        dst16[2 * i + 1] = kv_cache_e8_unpack_i4(src8[i], 1);
    }
}

template <typename Geometry>
__device__ __forceinline__ void kv_cache_append_full_e8_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, int token, int kv_head,
    int physical_page, int page_offset, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    const unsigned SubMask      = 0xFFu << (lane & 24);

    float values[8];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(k[kv_cache_e8_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
    const auto k_quant = kv_cache_e8_scale_params(warp_max(local_absmax, FullMask));

    // E8 NLP per subspace, then the documented half-coset rint into the 4-bit code range.
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d        = lane + 32 * r;
        const float x      = values[r] * k_quant.inverse_scale;
        const float x_e8   = e8_project_8d_warp_single(x, lane, SubMask);
        const std::int8_t code = kv_cache_e8_lattice_code(x_e8);
        const std::int8_t code_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(code), 1));
        if ((lane & 1) == 0) {
            cache_k[kv_cache_e8_i4_code_index<Geometry>(physical_page, kv_head, d / 2,
                                                        page_offset)] =
                kv_cache_e8_pack_i4(code, code_hi);
        }
    }
    if (lane == 0) {
        scale_k[kv_cache_e8_i4_scale_index<Geometry>(physical_page, kv_head, page_offset)] =
            k_quant.scale;
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(v[kv_cache_e8_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
    // V takes its own per-token scale (V-row absmax / 7), independent of the K scale.
    local_absmax = 0.0F;
#pragma unroll
    for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
    const auto v_quant = kv_cache_e8_scale_params(warp_max(local_absmax, FullMask));

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d    = lane + 32 * r;
        const std::int8_t code =
            kv_cache_e8_i4_quant_code(values[r], v_quant.inverse_scale);
        const std::int8_t code_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(code), 1));
        if ((lane & 1) == 0) {
            cache_v[kv_cache_e8_i4_code_index<Geometry>(physical_page, kv_head, d / 2,
                                                        page_offset)] =
                kv_cache_e8_pack_i4(code, code_hi);
        }
    }
    if (lane == 0) {
        scale_v[kv_cache_e8_i4_scale_index<Geometry>(physical_page, kv_head, page_offset)] =
            v_quant.scale;
    }
}


} // namespace ninfer::ops
