// Implements: include/ninfer/ops/tp2/argmax.h (argmax_split_pack, argmax_split_select)
// Match: validated contiguous BF16 [R,T] logits with R <= kArgmaxSplitMaxRowsPerRank, I32 [T]
// rows, BF16 [8,T] candidates.
// Algorithm assumptions: one thread per column; a candidate is four base-256 digits, each an
// integer BF16 represents exactly, so the summing exchange of the two ranks' disjoint digits is
// exact.
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/tp2/argmax.h"
#include "ops/launcher/argmax_split.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kSplitBlock = 128;

__device__ __forceinline__ __nv_bfloat16 digit(std::uint32_t value) {
    return __float2bfloat16_rn(static_cast<float>(value & 0xffU));
}

__device__ __forceinline__ std::uint32_t digit_value(__nv_bfloat16 digit) {
    return static_cast<std::uint32_t>(__bfloat162float(digit));
}

__global__ void argmax_split_pack_kernel(const __nv_bfloat16* logits, const std::int32_t* local,
                                         std::int32_t rows, std::int32_t columns, int rank,
                                         __nv_bfloat16* candidates) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= columns) { return; }
    // argmax() writes a row of [0, rows); the clamp only keeps a broken input inside the block.
    const std::int32_t row = min(max(local[t], 0), rows - 1);
    const std::uint32_t bits =
        __bfloat16_as_ushort(logits[static_cast<std::int64_t>(t) * rows + row]);
    const auto l      = static_cast<std::uint32_t>(row);
    __nv_bfloat16* out = candidates + static_cast<std::int64_t>(t) * kArgmaxSplitCandidateRows;
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
#pragma unroll
    for (int i = 0; i < kArgmaxSplitCandidateRows; ++i) { out[i] = zero; }
    out[4 * rank + 0] = digit(bits >> 8U);
    out[4 * rank + 1] = digit(bits);
    out[4 * rank + 2] = digit(l >> 8U);
    out[4 * rank + 3] = digit(l);
}

__global__ void argmax_split_select_kernel(const __nv_bfloat16* candidates,
                                           std::int32_t rows_per_rank, std::int32_t columns,
                                           std::int32_t* out) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= columns) { return; }
    const __nv_bfloat16* c = candidates + static_cast<std::int64_t>(t) * kArgmaxSplitCandidateRows;
    float value[2];
    std::int32_t row[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const auto bits = static_cast<unsigned short>((digit_value(c[4 * r + 0]) << 8U) |
                                                      digit_value(c[4 * r + 1]));
        value[r]        = __bfloat162float(__ushort_as_bfloat16(bits));
        row[r] = static_cast<std::int32_t>((digit_value(c[4 * r + 2]) << 8U) |
                                           digit_value(c[4 * r + 3]));
    }
    // Rank 1's rows all follow rank 0's, so only a strictly larger value moves the argmax there.
    out[t] = value[1] > value[0] ? rows_per_rank + row[1] : row[0];
}

} // namespace

void argmax_split_pack_launch(const Tensor& logits, const Tensor& local, int rank,
                              Tensor& candidates, cudaStream_t stream) {
    const std::int32_t columns = logits.ne[1];
    argmax_split_pack_kernel<<<static_cast<unsigned int>(div_up(columns, kSplitBlock)),
                               kSplitBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(local.data), logits.ne[0], columns, rank,
        static_cast<__nv_bfloat16*>(candidates.data));
    CUDA_CHECK(cudaGetLastError());
}

void argmax_split_select_launch(const Tensor& candidates, std::int32_t rows_per_rank, Tensor& out,
                                cudaStream_t stream) {
    const std::int32_t columns = candidates.ne[1];
    argmax_split_select_kernel<<<static_cast<unsigned int>(div_up(columns, kSplitBlock)),
                                 kSplitBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(candidates.data), rows_per_rank, columns,
        static_cast<std::int32_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
