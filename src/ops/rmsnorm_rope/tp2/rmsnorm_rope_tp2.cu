// Implements: include/ninfer/ops/tp2/rmsnorm_rope.h (rmsnorm_rope_head_block). Our file.
// Match: q BF16 [128,16,W,B], k BF16 [128,4,W,B], W=2..16, B=1..8.
// Algorithm assumptions: upstream's per-head device code (rmsnorm_rope/d128.cuh) and coefficient
// table (common/dflash_rope.cuh); only the head counts per token differ from the 32/8 kernel
// (rmsnorm_rope/kernel.cuh): two Q CTAs of eight heads and one K CTA of four per token.
#include "ninfer/ops/tp2/rmsnorm_rope.h"

#include "core/device.h"
#include "ops/common/dflash_rope.cuh"
#include "ops/rmsnorm_rope/d128.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim     = 128;
constexpr int kQueryHeads  = 16;
constexpr int kKeyHeads    = 4;
constexpr int kQueryCtas   = kQueryHeads / 8;

__global__ __launch_bounds__(256) void rmsnorm_rope_d128_head_block_kernel(
    const std::int32_t* __restrict__ positions, const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k) {
    constexpr int kPairs = 64;
    const int token      = blockIdx.x;
    const bool query     = blockIdx.y < kQueryCtas;
    const int head =
        (query ? static_cast<int>(blockIdx.y) * 8 : 0) + static_cast<int>(threadIdx.x) / 32;
    const int lane     = threadIdx.x % 32;
    auto* data         = reinterpret_cast<__nv_bfloat162*>(query ? q : k);
    const auto* weight = reinterpret_cast<const __nv_bfloat162*>(query ? q_norm : k_norm);
    __shared__ float cos_cache[kPairs];
    __shared__ float sin_cache[kPairs];
    __shared__ __nv_bfloat162 weight_cache[kPairs];
    if (threadIdx.x < kPairs) {
        const int pair = threadIdx.x;
        dflash_rope_sincos(positions, token, pair, &sin_cache[pair], &cos_cache[pair]);
        weight_cache[pair] = weight[pair];
    }
    __syncthreads();
    if (!query && head >= kKeyHeads) { return; }
    const std::int64_t base =
        (static_cast<std::int64_t>(token) * (query ? kQueryHeads : kKeyHeads) + head) * kPairs;
    const auto out = detail::rmsnorm_rope_d128_head(data[base + lane], data[base + lane + 32],
                                                    weight_cache[lane], weight_cache[lane + 32],
                                                    cos_cache, sin_cache, lane);
    data[base + lane]      = out.first;
    data[base + lane + 32] = out.second;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

void require(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2,
             std::int32_t n3, const char* label) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3 ||
        !t.is_contiguous() || t.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(t.data) & 3U) != 0) {
        throw std::invalid_argument(std::string("rmsnorm_rope_head_block: invalid ") + label);
    }
}

} // namespace

void rmsnorm_rope_head_block(const Tensor& positions, const Tensor& q_norm_weight,
                             const Tensor& k_norm_weight, Tensor& q, Tensor& k,
                             cudaStream_t stream) {
    const std::int32_t width = q.ne[2];
    const std::int32_t batch = q.ne[3];
    if (width < 2 || width > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument("rmsnorm_rope_head_block: W must be 2..16 and B 1..8");
    }
    require(q, DType::BF16, kHeadDim, kQueryHeads, width, batch, "q");
    require(k, DType::BF16, kHeadDim, kKeyHeads, width, batch, "k");
    require(q_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, "q norm weight");
    require(k_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, "k norm weight");
    require(positions, DType::I32, width, batch, 1, 1, "positions");
    if (overlaps(q, k) || overlaps(q, positions) || overlaps(k, positions) ||
        overlaps(q, q_norm_weight) || overlaps(q, k_norm_weight) || overlaps(k, q_norm_weight) ||
        overlaps(k, k_norm_weight)) {
        throw std::invalid_argument("rmsnorm_rope_head_block: mutable tensors overlap");
    }
    const dim3 grid(static_cast<unsigned>(width * batch), kQueryCtas + 1);
    rmsnorm_rope_d128_head_block_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const __nv_bfloat16*>(q_norm_weight.data),
        static_cast<const __nv_bfloat16*>(k_norm_weight.data),
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
