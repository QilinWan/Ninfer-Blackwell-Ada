#pragma once

// E8 Conway-Sloane (rk4v4-e8) causal prompt kernel for the registered head geometries.
// Ada-only route. Q and cached K use the same fixed register-only D256 rotation; K keeps its
// exact-int8 E8 lattice codes (absmax/7 code domain) so QK stays on m16n8k32.s8 Tensor Cores
// with one per-row/per-key scale. The 4-bit V codes are decoded with packed FP16 arithmetic
// while producer warps execute QK. Sixteen warps split each 16-row FP16 PV output across
// four 64-dimension slices. The arena holds K codes, V codes, K i8 (decoded), and V FP16
// (decoded): the same 4*Bc*D bytes as the int8 arena.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kv_cache/e8/e8_i4_code.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptE8Warps      = 16;
inline constexpr int kCausalPromptE8Threads    = kCausalPromptE8Warps * 32;
inline constexpr int kCausalPromptE8Br         = 64;
inline constexpr int kCausalPromptE8Bc         = 64;
inline constexpr int kCausalPromptE8Groups     = 1;
inline constexpr int kCausalPromptE8DB16       = kCausalPromptHeadDim / 2;
inline constexpr int kCausalPromptE8CodeExtent = kCausalPromptHeadDim / 2;
inline constexpr int kCausalPromptE8RowTiles   = kCausalPromptE8Br / 16;
inline constexpr int kCausalPromptE8DConsumers = kCausalPromptE8Warps / kCausalPromptE8RowTiles;

inline constexpr int kCausalPromptE8QBytes = kCausalPromptE8Br * kCausalPromptHeadDim;
inline constexpr int kCausalPromptE8QScaleBytes =
    kCausalPromptE8Br * kCausalPromptE8Groups * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptE8KBytes = kCausalPromptE8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptE8CodeBytes =
    kCausalPromptE8Bc * kCausalPromptE8CodeExtent;
inline constexpr int kCausalPromptE8VStageBytes =
    kCausalPromptE8Bc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptE8PBytes =
    kCausalPromptE8Br * kCausalPromptE8Bc * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptE8ScaleBytes =
    2 * kCausalPromptE8Bc * kCausalPromptE8Groups * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptE8StatsBytes =
    2 * kCausalPromptE8Br * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptE8SmemBytes =
    kCausalPromptE8QBytes + kCausalPromptE8QScaleBytes + 2 * kCausalPromptE8CodeBytes +
    kCausalPromptE8KBytes + kCausalPromptE8VStageBytes + kCausalPromptE8PBytes +
    kCausalPromptE8ScaleBytes + kCausalPromptE8StatsBytes;

static_assert(kCausalPromptE8Groups == 1);
static_assert(kCausalPromptE8DConsumers == 4);
static_assert(kCausalPromptE8SmemBytes == 91136);

// Eight 4-bit V codes (four packed bytes) decoded to eight FP16 values scaled by one
// per-key FP16 scale.
__device__ __forceinline__ int4 causal_prompt_e8_dequant_f16x8(const std::uint8_t* bytes4,
                                                               __half scale) {
    const int2 raw       = load_vec<int2>(bytes4);
    const auto* p        = reinterpret_cast<const std::uint8_t*>(&raw);
    const __half2 s2     = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // Byte i packs codes 2i (low nibble, even dim) and 2i + 1 (high nibble, odd dim);
        // both nibbles of all four bytes are consumed, so the int2 load covers every dim.
        const __half2 code2 = __floats2half2_rn(
            static_cast<float>(kv_cache_e8_unpack_i4(p[i], 0)),
            static_cast<float>(kv_cache_e8_unpack_i4(p[i], 1)));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

template <typename Geometry, typename Metadata>
__global__ __maxnreg__(120) void causal_attention_prompt_e8_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = kCausalPromptE8Br;
    constexpr int Bc            = kCausalPromptE8Bc;
    constexpr int DB16          = kCausalPromptE8DB16;
    constexpr int CodeExtent    = kCausalPromptE8CodeExtent;
    constexpr int Groups        = kCausalPromptE8Groups;
    constexpr int GroupKc       = D / 32 / Groups;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / (kCausalPromptE8DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kCausalPromptE8RowTiles;
    constexpr int VWorkerWarps  = kCausalPromptE8Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 8);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale       = reinterpret_cast<float*>(q_i8 + kCausalPromptE8QBytes);
    std::int8_t* k_code  = reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                                      kCausalPromptE8QScaleBytes);
    std::int8_t* v_code  = k_code + kCausalPromptE8CodeBytes;
    std::int8_t* k_i8    = v_code + kCausalPromptE8CodeBytes;
    __half* v_f16        = reinterpret_cast<__half*>(k_i8 + kCausalPromptE8KBytes);
    __half* p_s          = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) +
                                                     kCausalPromptE8VStageBytes);
    __half* k_scale_s    =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kCausalPromptE8PBytes);
    __half* v_scale_s    = k_scale_s + Bc * Groups;
    float* alpha_s       = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s     = alpha_s + Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 kCausalPromptE8Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Quantize Q cooperatively. One full warp rotates and encodes one D256 row at a time.
    for (int row = warp; row < Br; row += kCausalPromptE8Warps) {
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = 0.0f;
            if (row < tile_rows) {
                q_values[r] =
                    __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)]);
            }
        }
        normalized_hadamard_d256_inplace(q_values, lane);

        // One per-token scale for the whole rotated row (E8 code domain, absmax/7); the codes
        // are exact int8 values in the +/-7 range.
        float absmax = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) { absmax = fmaxf(absmax, fabsf(q_values[r])); }
        absmax          = warp_max(absmax, FullMask);
        const float qs  = absmax > 0.0f ? absmax / kE8I4ScaleRange : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            causal_prompt_store_byte_swizzled(q_i8, row, d, kv_cache_int8_quant_code(q_values[r], inv));
        }
        if (lane == 0) { q_scale[row * Groups] = qs; }
    }
    __syncthreads();

    // Stage one key block's per-key scales (4 keys per 8-byte transfer: the scale plane is one
    // FP16 per key, contiguous in page-offset order) and the K/V 4-bit code planes
    // (16 bytes = 32 codes = 32 dimensions per chunk).
    auto issue_kv_tile = [&](int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int q4 = tid; q4 * 4 < Bc; q4 += kCausalPromptE8Threads) {
            const int key = tile_k0 + 4 * q4;
            __half* kd    = &k_scale_s[4 * q4];
            __half* vd    = &v_scale_s[4 * q4];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    kv_cache_e8_i4_scale_index<Geometry>(physical_page, kv_head, 4 * q4);
                ninfer::ops::cp_async<8>(kd, &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(vd, &cache_v_scale[off]);
            } else {
                store_vec(reinterpret_cast<int*>(kd), make_int2(0, 0));
                store_vec(reinterpret_cast<int*>(vd), make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (CodeExtent / 16); chunk += kCausalPromptE8Threads) {
            const int key_l = chunk / (CodeExtent / 16);
            const int dc    = chunk - key_l * (CodeExtent / 16);
            const int d     = dc * 32;
            const int key   = tile_k0 + key_l;
            if (key <= max_query_abs) {
                const std::int64_t off = kv_cache_e8_i4_code_index<Geometry>(
                    physical_page, kv_head, d / 2, key_l);
                cp_async<16, Cache::cg>(&k_code[key_l * CodeExtent + dc * 16], &cache_k[off]);
                cp_async<16, Cache::cg>(&v_code[key_l * CodeExtent + dc * 16], &cache_v[off]);
            } else {
                store_vec(&k_code[key_l * CodeExtent + dc * 16], make_int4(0, 0, 0, 0));
                store_vec(&v_code[key_l * CodeExtent + dc * 16], make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    // Unpack the staged K codes into the swizzled int8 QK tile. K codes are exact int8 values
    // (the E8 lattice code domain is [-8, 7]); the per-key scale is applied in the QK rescale.
    auto decode_k_tile = [&]() {
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (CodeExtent / 16); chunk += kCausalPromptE8Threads) {
            const int key_l = chunk / (CodeExtent / 16);
            const int dc    = chunk - key_l * (CodeExtent / 16);
            const int4 raw  = load_vec<int4>(&k_code[key_l * CodeExtent + dc * 16]);
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int d = dc * 32 + 2 * i;
                const std::uint8_t packed = bytes[i];
                causal_prompt_store_byte_swizzled(k_i8, key_l, d, kv_cache_e8_unpack_i4(packed, 0));
                causal_prompt_store_byte_swizzled(k_i8, key_l, d + 1,
                                                  kv_cache_e8_unpack_i4(packed, 1));
            }
        }
        __syncthreads();
    };

    issue_kv_tile(0);
    ninfer::ops::cp_wait<0>();
    decode_k_tile();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0[Groups - 1 > 0 ? Groups - 1 : 1];
    float q_scale_r1[Groups - 1 > 0 ? Groups - 1 : 1];
    if (warp < ProducerWarps) {
        const int scale_row0 = warp * 16 + gid;
        const int scale_row1 = scale_row0 + 8;
        float qs0 = lid == 0 ? q_scale[scale_row0 * Groups] : 0.0f;
        float qs1 = lid == 0 ? q_scale[scale_row1 * Groups] : 0.0f;
        q_scale_r0[0] = __shfl_sync(FullMask, qs0, gid * 4);
        q_scale_r1[0] = __shfl_sync(FullMask, qs1, gid * 4);
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;
        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

            unsigned af[GroupKc][4];
#pragma unroll
            for (int kk = 0; kk < GroupKc; ++kk) {
                const int acol = kk * 16 + a_coloff;
                ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(row_base + a_rowoff, acol)]));
            }

#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int brow = nt * 8 + b_rin;
                    const int bcol = kk * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                           bf[1]);
                }
                const int keya = nt * 8 + 2 * lid;
                const int keyb = keya + 1;
                float ks0      = 0.0f;
                float ks1      = 0.0f;
                if (gid == 0) {
                    ks0 = __half2float(k_scale_s[keya * Groups]);
                    ks1 = __half2float(k_scale_s[keyb * Groups]);
                }
                ks0          = __shfl_sync(FullMask, ks0, lid);
                ks1          = __shfl_sync(FullMask, ks1, lid);
                score[nt][0] = __fmaf_rn(q_scale_r0[0] * ks0, static_cast<float>(c0), score[nt][0]);
                score[nt][1] = __fmaf_rn(q_scale_r0[0] * ks1, static_cast<float>(c1), score[nt][1]);
                score[nt][2] = __fmaf_rn(q_scale_r1[0] * ks0, static_cast<float>(c2), score[nt][2]);
                score[nt][3] = __fmaf_rn(q_scale_r1[0] * ks1, static_cast<float>(c3), score[nt][3]);
            }

            const int row0   = row_base + gid;
            const int row1   = row0 + 8;
            const int qabs0  = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1  = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col0)] = __float2half_rn(p00);
                p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col1)] = __float2half_rn(p01);
                p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col0)] = __float2half_rn(p10);
                p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col1)] = __float2half_rn(p11);
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
                if (key <= max_query_abs) {
                    // One per-key scale for the whole row; each 8-lane subgroup's leader
                    // loads it, then broadcasts within the warp.
                    __half vs = __float2half_rn(0.0f);
                    if ((lane & 7) == 0) { vs = v_scale_s[key_l * Groups]; }
                    vs = __shfl_sync(FullMask, __half2float(vs), (lane & 7) * 8);
                    vs = __float2half_rn(__half2float(vs));
                    store_vec(dst, causal_prompt_e8_dequant_f16x8(
                                         reinterpret_cast<const std::uint8_t*>(
                                             &v_code[key_l * CodeExtent + (d >> 1)]),
                                         vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) { issue_kv_tile((kb + 1) * Bc); }

        const int row_tile = warp % kCausalPromptE8RowTiles;
        const int d_slice  = warp / kCausalPromptE8RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       causal_prompt_p_swz<Bc>(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
        }
        if (has_next) {
            ninfer::ops::cp_wait<0>();
            decode_k_tile();
        }
    }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kCausalPromptE8RowTiles;
    const int d_slice  = warp / kCausalPromptE8RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             kCausalPromptE8Threads);
}

} // namespace ninfer::ops
