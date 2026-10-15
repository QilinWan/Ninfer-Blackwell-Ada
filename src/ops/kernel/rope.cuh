#pragma once

// Implements: include/ninfer/ops/rope.h
// Fixed matches: BF16 Qwen3.6 Text 24Q/4K and 16Q/2K at D/R=256/64, DFlash 32Q/8K at
// D/R=128/128, plus packed Vision 16Q/16K at D/R=72/72. One CTA owns one token and shares its
// rotary coefficients across heads.

#include "ninfer/ops/rope.h"
#include "ops/common/dflash_rope.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

enum class RopeKernelMode : std::int32_t {
    Text1D,
    DflashText1D,
    TextMrope,
    Vision2D,
};

inline constexpr int kRopeMaxHalf = 128;

// Double-precision inverse frequencies, as for the DFlash route (ops/common/dflash_rope.cuh): the
// rotary phase is formed in double, so the coefficient table has to carry double precision too.
// Each entry is the correctly rounded value of theta^(-2*pairs/exp-div), and every entry projects
// onto exactly the float constant this table replaces (verified digit by digit), so the float
// behaviour of short-context routes is unchanged.
static __device__ __constant__ double kTextRopeInvFrequency[32] = {
      1.00000000000000000e+00,  0.604296390238132819e+00,  0.365174127254837706e+00,
      0.220673406908458980e+00,  0.133352143216332403e+00,  0.0805842187761481817e+00,
      0.0486967525165863115e+00,  0.0294272717620928181e+00,  0.0177827941003892280e+00,
      0.0107460782832131750e+00,  0.00649381631576211315e+00,  0.00392418975848453586e+00,
      0.00237137370566165526e+00,  0.00143301257023696274e+00,  0.000865964323360065352e+00,
      0.000523299114681494688e+00,  0.000316227766016837933e+00,  0.000191095297497044052e+00,
      0.000115478198468945818e+00,  0.0000697830584859866338e+00,  0.0000421696503428582249e+00,
      0.0000254829674797934653e+00,  0.0000153992652605949199e+00,  0.00000930572040929698979e+00,
      0.00000562341325190349080e+00,  0.00000339820832894255937e+00,  0.00000205352502645714607e+00,
      0.00000124093776075171957e+00,  0.000000749894209332455827e+00,  0.000000453158363760081788e+00,
      0.000000273841963426436129e+00,  0.000000165481709994318142e+00,
};

static __device__ __constant__ double kVisionRopeInvFrequency[18] = {
      1.00000000000000000e+00,  0.599484250318941015e+00,  0.359381366380462730e+00,
      0.215443469003188372e+00,  0.129154966501488388e+00,  0.0774263682681127060e+00,
      0.0464158883361277889e+00,  0.0278255940220712460e+00,  0.0166810053720005875e+00,
      0.0100000000000000000e+00,  0.00599484250318941015e+00,  0.00359381366380462730e+00,
      0.00215443469003188372e+00,  0.00129154966501488388e+00,  0.000774263682681127060e+00,
      0.000464158883361277889e+00,  0.000278255940220712460e+00,  0.000166810053720005875e+00,
};

template <RopeKernelMode Mode>
__device__ __forceinline__ void fixed_axis_frequency(int pair, int* axis, double* frequency) {
    if constexpr (Mode == RopeKernelMode::Vision2D) {
        *axis      = pair / 18;
        *frequency = kVisionRopeInvFrequency[pair % 18];
    } else {
        *axis      = Mode == RopeKernelMode::TextMrope ? pair % 3 : 0;
        *frequency = kTextRopeInvFrequency[pair];
    }
}

// Form the phase and its 2π reduction in double, then hand a value inside [-π, π] to the accurate
// float sincos, exactly as the DFlash route does. The rotary phase grows linearly with the position,
// so a float phase carries argument-reduction error of the same order as the profile a rope output
// is checked against: at position 1,048,000 the float ulp of a 3.8e5 rad phase is 0.031 rad, four
// times the BF16 pair tolerance a long-context tail must stay inside. One CTA computes this kHalf
// times and shares the coefficients across heads, so the double work is off the bandwidth path.
__device__ __forceinline__ void reduced_sincos(std::int32_t position, double frequency, float* sine,
                                               float* cosine) {
    constexpr double kInvTwoPi = 1.59154943091895336e-01;
    constexpr double kTwoPi    = 6.28318530717958648e+00;
    const double angle         = static_cast<double>(position) * frequency;
    const double turns         = angle * kInvTwoPi;
    const float reduced        = static_cast<float>(angle - nearbyint(turns) * kTwoPi);
    sincosf(reduced, sine, cosine);
}

template <RopeKernelMode Mode>
__device__ __forceinline__ void fixed_sincos(const std::int32_t* positions, int tokens, int token,
                                             int pair, float* sine, float* cosine) {
    if constexpr (Mode == RopeKernelMode::DflashText1D) {
        dflash_rope_sincos(positions, token, pair, sine, cosine);
    } else {
        int axis  = 0;
        double frequency;
        fixed_axis_frequency<Mode>(pair, &axis, &frequency);
        reduced_sincos(positions[static_cast<std::int64_t>(axis) * tokens + token], frequency, sine,
                       cosine);
    }
}

template <int HeadDim, int Half>
__device__ __forceinline__ void apply_rope_head(__nv_bfloat16* data, std::int64_t token_stride,
                                                int head, int token, int lane, float c0, float c1,
                                                float s0, float s1) {
    constexpr int kHalfPair = Half / 2;
    if (lane >= kHalfPair) { return; }
    const std::int64_t base =
        static_cast<std::int64_t>(token) * token_stride + static_cast<std::int64_t>(head) * HeadDim;
    auto* data2         = reinterpret_cast<__nv_bfloat162*>(data + base);
    const float2 first  = __bfloat1622float2(data2[lane]);
    const float2 second = __bfloat1622float2(data2[lane + kHalfPair]);
    data2[lane] = __floats2bfloat162_rn(first.x * c0 - second.x * s0, first.y * c1 - second.y * s1);
    data2[lane + kHalfPair] =
        __floats2bfloat162_rn(second.x * c0 + first.x * s0, second.y * c1 + first.y * s1);
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
__global__ void rope_fixed_kernel(const std::int32_t* positions, __nv_bfloat16* q, __nv_bfloat16* k,
                                  std::int32_t tokens, std::int64_t q_token_stride,
                                  std::int64_t k_token_stride) {
    constexpr int kHeadDim = Mode == RopeKernelMode::Vision2D       ? 72
                             : Mode == RopeKernelMode::DflashText1D ? 128
                                                                    : 256;
    constexpr int kHalf    = Mode == RopeKernelMode::Vision2D       ? 36
                             : Mode == RopeKernelMode::DflashText1D ? 64
                                                                    : 32;
    const int token        = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }

    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];
    if (threadIdx.x < kHalf) {
        const int pair = static_cast<int>(threadIdx.x);
        fixed_sincos<Mode>(positions, tokens, token, pair, &sin_cache[pair], &cos_cache[pair]);
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    float c0 = 0.0F, c1 = 0.0F, s0 = 0.0F, s1 = 0.0F;
    if (lane < kHalf / 2) {
        const int pair = lane * 2;
        c0             = cos_cache[pair];
        c1             = cos_cache[pair + 1];
        s0             = sin_cache[pair];
        s1             = sin_cache[pair + 1];
    }
    for (int combined_head = warp; combined_head < QHeads + KHeads; combined_head += block_warps) {
        if (combined_head < QHeads) {
            apply_rope_head<kHeadDim, kHalf>(q, q_token_stride, combined_head, token, lane, c0, c1,
                                             s0, s1);
        } else {
            apply_rope_head<kHeadDim, kHalf>(k, k_token_stride, combined_head - QHeads, token, lane,
                                             c0, c1, s0, s1);
        }
    }
}

template <RopeKernelMode Mode, int QHeads, int KHeads, int HeadsPerBlock>
__global__ void rope_fixed_split_kernel(const std::int32_t* positions, __nv_bfloat16* q,
                                        __nv_bfloat16* k, std::int32_t tokens,
                                        std::int64_t q_token_stride, std::int64_t k_token_stride) {
    static_assert(Mode == RopeKernelMode::DflashText1D);
    constexpr int kHeadDim       = 128;
    constexpr int kHalf          = 64;
    constexpr int kCombinedHeads = QHeads + KHeads;
    constexpr int kHeadGroups    = (kCombinedHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    const int token              = static_cast<int>(blockIdx.x) / kHeadGroups;
    const int head_group         = static_cast<int>(blockIdx.x) % kHeadGroups;
    if (token >= tokens) { return; }

    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];
    if (threadIdx.x < kHalf) {
        const int pair = static_cast<int>(threadIdx.x);
        fixed_sincos<Mode>(positions, tokens, token, pair, &sin_cache[pair], &cos_cache[pair]);
    }
    __syncthreads();

    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int local_head    = static_cast<int>(threadIdx.x) >> 5;
    const int combined_head = head_group * HeadsPerBlock + local_head;
    if (local_head >= HeadsPerBlock || combined_head >= kCombinedHeads) { return; }
    const int pair = lane * 2;
    const float c0 = cos_cache[pair];
    const float c1 = cos_cache[pair + 1];
    const float s0 = sin_cache[pair];
    const float s1 = sin_cache[pair + 1];
    if (combined_head < QHeads) {
        apply_rope_head<kHeadDim, kHalf>(q, q_token_stride, combined_head, token, lane, c0, c1, s0,
                                         s1);
    } else {
        apply_rope_head<kHeadDim, kHalf>(k, k_token_stride, combined_head - QHeads, token, lane, c0,
                                         c1, s0, s1);
    }
}

__device__ __forceinline__ void generic_axis_frequency(int axes, int head_dim, int rotary_dim,
                                                       int pair, int* axis, double* exponent) {
    if (axes == 2 && head_dim == 72 && rotary_dim == 72) {
        *axis           = pair / 18;
        const int local = pair % 18;
        *exponent       = -2.0 * static_cast<double>(local) / 36.0;
    } else {
        *axis     = axes == 3 ? pair % 3 : 0;
        *exponent = -2.0 * static_cast<double>(pair) / static_cast<double>(rotary_dim);
    }
}

static __global__ void rope_generic_kernel(const std::int32_t* positions, std::int32_t axes,
                                           __nv_bfloat16* q, __nv_bfloat16* k,
                                           std::int32_t head_dim, std::int32_t rotary_dim,
                                           float theta, std::int32_t q_heads, std::int32_t k_heads,
                                           std::int32_t tokens, std::int64_t q_token_stride,
                                           std::int64_t k_token_stride, TextRopeScaling scaling) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }
    const int half = rotary_dim / 2;
    __shared__ float cos_cache[kRopeMaxHalf];
    __shared__ float sin_cache[kRopeMaxHalf];
    if (threadIdx.x < static_cast<unsigned>(half)) {
        const int pair = static_cast<int>(threadIdx.x);
        if (axes == 1 && head_dim == 128 && rotary_dim == 128 && theta == 1.0e7F) {
            fixed_sincos<RopeKernelMode::DflashText1D>(positions, tokens, token, pair,
                                                       &sin_cache[pair], &cos_cache[pair]);
        } else {
            int axis      = 0;
            double exponent = 0.0;
            generic_axis_frequency(axes, head_dim, rotary_dim, pair, &axis, &exponent);
            // Native frequencies come from the double power, so a shape outside the fixed tables
            // keeps the same phase accuracy as the selected routes.
            const double frequency = scaling.enabled
                                         ? static_cast<double>(scaling.inverse_frequency[pair])
                                         : pow(static_cast<double>(theta), exponent);
            const std::int32_t position =
                positions[static_cast<std::int64_t>(axis) * tokens + token];
            reduced_sincos(position, frequency, &sin_cache[pair], &cos_cache[pair]);
            if (scaling.enabled) {
                sin_cache[pair] *= scaling.attention_factor;
                cos_cache[pair] *= scaling.attention_factor;
            }
        }
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    for (int combined_head = warp; combined_head < q_heads + k_heads;
         combined_head += block_warps) {
        const bool is_q             = combined_head < q_heads;
        const int head              = is_q ? combined_head : combined_head - q_heads;
        __nv_bfloat16* data         = is_q ? q : k;
        const std::int64_t stride_t = is_q ? q_token_stride : k_token_stride;
        const std::int64_t base     = static_cast<std::int64_t>(token) * stride_t +
                                  static_cast<std::int64_t>(head) * head_dim;
        for (int pair = lane; pair < half; pair += 32) {
            const float first        = __bfloat162float(data[base + pair]);
            const float second       = __bfloat162float(data[base + pair + half]);
            const float c            = cos_cache[pair];
            const float s            = sin_cache[pair];
            data[base + pair]        = __float2bfloat16_rn(first * c - second * s);
            data[base + pair + half] = __float2bfloat16_rn(second * c + first * s);
        }
    }
}

} // namespace ninfer::ops
