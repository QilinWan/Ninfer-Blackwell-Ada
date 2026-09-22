#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_geometry.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Routes parent rows [0,Q+K+V) to qkv and the trailing Z rows to z. KeyRows is shared by the Q and
// K sections and ValueRows by the V and Z sections.
template <std::int32_t KeyRows, std::int32_t ValueRows>
struct Fp8GdnInputSectionOutput {
    static constexpr std::int32_t kQueryRows = KeyRows;
    static constexpr std::int32_t kKeyRows   = KeyRows;
    static constexpr std::int32_t kValueRows = ValueRows;
    static constexpr std::int32_t kQkvRows   = kQueryRows + kKeyRows + kValueRows;
    static constexpr std::int32_t kZRows     = ValueRows;
    static constexpr std::int32_t kRows      = kQkvRows + kZRows;

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

using Fp8GdnInputOutput = Fp8GdnInputSectionOutput<2048, 6144>;

// Two-device shard: each rank's contiguous [8192,5120] parent holds its 8 of 16 key heads and 24
// of 48 value heads in the same Q|K|V|Z order.
using Fp8GdnInputShardOutput = Fp8GdnInputSectionOutput<1024, 3072>;

static_assert(Fp8GdnInputOutput::kRows == 16384);
static_assert((Fp8GdnInputOutput::kQkvRows % 128) == 0);
static_assert((Fp8GdnInputOutput::kZRows % 128) == 0);
static_assert(Fp8GdnInputShardOutput::kRows == 8192);
static_assert((Fp8GdnInputShardOutput::kQkvRows % 128) == 0);
static_assert((Fp8GdnInputShardOutput::kZRows % 128) == 0);

} // namespace ninfer::ops::detail
