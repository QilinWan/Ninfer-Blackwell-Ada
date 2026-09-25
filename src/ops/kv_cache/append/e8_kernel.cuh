#pragma once

// E8 Conway-Sloane lattice K / i4 V append kernel (Ada-only KV route). K rows receive the
// fixed FP32 D256 Hadamard rotation, are E8-projected per 8-dim subspace, and the projected
// lattice points are stored as 4-bit codes in the absmax/7 domain; V rows receive the same
// rotation and are stored as plain 4-bit codes. Both rows carry one per-token FP16 scale.
//
// Warp register map: lane l holds dimensions l + 32r (r = 0..7), so 8-lane subgroup g owns
// subspace 4r + g of each register row - exactly the layout e8_project_8d_warp_single expects
// with sub_mask 0xFF << (lane & 24).

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/e8/e8_i4_code.cuh"
#include "ops/kv_cache/e8/e8_lattice.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_e8_kernel(const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v,
                                        const std::int32_t* __restrict__ positions,
                                        Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                        std::uint8_t* __restrict__ cache_v,
                                        __half* __restrict__ scale_k,
                                        __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens = metadata.valid_tokens(width);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int unit   = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units  = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_e8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token,
                                          kv_head, physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void kv_cache_append_full_e8_page_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens        = metadata.valid_tokens(width);
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int kv_head       = static_cast<int>(blockIdx.y);
    const int base_position = positions[0];
    const int tile_position =
        (base_position / TokensPerTile + static_cast<int>(blockIdx.x)) * TokensPerTile;
    const int token_begin = max(0, tile_position - base_position);
    const int token_end   = min(tokens, tile_position + TokensPerTile - base_position);
    const int token       = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page  = lane == 0 ? block_table[tile_position >> kPagedKVPageShift] : 0;
    physical_page      = __shfl_sync(FullMask, physical_page, 0);
    const int position = base_position + token;
    kv_cache_append_full_e8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token,
                                          kv_head, physical_page, position & kPagedKVPageMask, lane);
}

} // namespace ninfer::ops
