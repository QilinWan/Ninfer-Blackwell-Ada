#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_geometry.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kFp8AttnInputQueryRows = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyRows   = 1024;
inline constexpr std::int32_t kFp8AttnInputGateRows  = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyBegin  = kFp8AttnInputQueryRows;
inline constexpr std::int32_t kFp8AttnInputGateBegin = kFp8AttnInputKeyBegin + kFp8AttnInputKeyRows;
inline constexpr std::int32_t kFp8AttnInputValueBegin =
    kFp8AttnInputGateBegin + kFp8AttnInputGateRows;

static_assert((kFp8AttnInputQueryRows % 8) == 0);
static_assert((kFp8AttnInputKeyRows % 8) == 0);
static_assert((kFp8AttnInputGateRows % 8) == 0);

// Stores one fused query|key|gate|value parent row into its independent section output. Gate has
// the query's row count and value has the key's.
template <std::int32_t QueryRows, std::int32_t KeyRows>
struct Fp8AttentionInputSections {
    static constexpr std::int32_t kQueryRows  = QueryRows;
    static constexpr std::int32_t kKeyRows    = KeyRows;
    static constexpr std::int32_t kGateRows   = QueryRows;
    static constexpr std::int32_t kKeyBegin   = kQueryRows;
    static constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
    static constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;
    static constexpr std::int32_t kParentRows = kValueBegin + kKeyRows;

    static_assert((kQueryRows % 8) == 0);
    static_assert((kKeyRows % 8) == 0);

    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        *destination(parent_row, token) = __float2bfloat16_rn(result);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

// The section output of the whole [14336,5120] fused parent.
using Fp8AttentionInputOutput = Fp8AttentionInputSections<kFp8AttnInputQueryRows,
                                                           kFp8AttnInputKeyRows>;

// A registered fused parent: its FP8 problem geometry and its section output.
template <class GeometryType, std::int32_t QueryRows, std::int32_t KeyRows>
struct Fp8AttnInputProblem {
    using Geometry = GeometryType;
    using Output   = Fp8AttentionInputSections<QueryRows, KeyRows>;

    static_assert(Geometry::kOutputRows == Output::kParentRows);
};

// The whole [14336,5120] parent, and one device's [7168,5120] shard under two-device tensor
// parallelism. The shard is a standalone weight whose sections are that device's head-local halves
// of the parent's sections (12 of 24 query/gate heads, 2 of 4 key/value heads), in the same
// query|key|gate|value order. It is never a view into the parent's payload.
using Fp8AttnInputParent = Fp8AttnInputProblem<Fp8N14336K5120, 6144, 1024>;
using Fp8AttnInputShard  = Fp8AttnInputProblem<Fp8N7168K5120, 3072, 512>;

// Calls `body.template operator()<Problem>()` for the registered problem with `parent_rows` rows.
template <class Body>
void visit_fp8_attn_input_problem(std::int32_t parent_rows, Body&& body) {
    if (parent_rows == Fp8AttnInputParent::Geometry::kOutputRows) {
        body.template operator()<Fp8AttnInputParent>();
        return;
    }
    if (parent_rows == Fp8AttnInputShard::Geometry::kOutputRows) {
        body.template operator()<Fp8AttnInputShard>();
        return;
    }
    throw std::invalid_argument("fp8 attn_input_proj: unsupported parent rows");
}

} // namespace ninfer::ops::detail
