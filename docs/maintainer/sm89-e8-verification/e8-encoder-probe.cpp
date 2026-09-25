// E8 fill probe #2: T>1 isolation.
//
// Probe #1 filled a single key (T=1, kv_heads=4) through the standalone append op and matched
// the host oracle bit for bit.  The softmax-attention suite fails rk4v4-e8 for T>1, so this
// probe repeats the same comparison with T=6 and prints the FIRST differing code byte, its
// decoded nibbles, the dim it belongs to, and the two raw bytes around it.
#include "ninfer/ops/kv_cache_append.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

constexpr int D    = 256;
constexpr int KVH  = 4;
constexpr int T    = 6;
constexpr int PAGE = 64;

void hadamard256(std::array<float, D>& v) {
    for (int stride = 1; stride < D; stride *= 2)
        for (int base = 0; base < D; base += 2 * stride)
            for (int off = 0; off < stride; ++off) {
                const float lo = v[base + off], hi = v[base + off + stride];
                v[base + off]          = lo + hi;
                v[base + off + stride] = lo - hi;
            }
    for (float& x : v) x *= 0x1p-4f;
}

void e8_project_host(const float x[8], float out[8]) {
    float f[8]; int sf = 0; float me = -1; int wd = 0;
    for (int i = 0; i < 8; ++i) {
        f[i] = std::nearbyint(x[i]); sf += int(f[i]);
        const float e = std::fabs(x[i] - f[i]);
        if (e > me) { me = e; wd = i; }
    }
    float d8[8];
    for (int i = 0; i < 8; ++i) d8[i] = f[i];
    if (sf & 1) d8[wd] += (x[wd] >= f[wd]) ? 1.f : -1.f;
    float fs[8]; int ss = 0; float mes = -1; int wds = 0;
    for (int i = 0; i < 8; ++i) {
        const float xs = x[i] - 0.5f;
        fs[i] = std::nearbyint(xs); ss += int(fs[i]);
        const float e = std::fabs(xs - fs[i]);
        if (e > mes) { mes = e; wds = i; }
    }
    float c1[8];
    for (int i = 0; i < 8; ++i) c1[i] = fs[i] + 0.5f;
    if (ss & 1) c1[wds] += ((x[wds] - 0.5f) >= fs[wds]) ? 1.f : -1.f;
    float dd = 0, dc = 0;
    for (int i = 0; i < 8; ++i) {
        dd += (x[i] - d8[i]) * (x[i] - d8[i]);
        dc += (x[i] - c1[i]) * (x[i] - c1[i]);
    }
    for (int i = 0; i < 8; ++i) out[i] = (dd <= dc) ? d8[i] : c1[i];
}

float f32_to_f16_bits(float x) {
    const __half h = __float2half_rn(x);
    return float(*reinterpret_cast<const std::uint16_t*>(&h));
}
float f16_bits_to_f32(float bits) {
    std::uint16_t b = (std::uint16_t)bits;
    return __half2float(*reinterpret_cast<const __half*>(&b));
}

// Returns the 128 packed code bytes for one row (already BF16-rounded input).
void oracle_codes(const std::array<float, D>& src, bool lattice,
                  std::vector<std::uint8_t>& codes, std::uint16_t& scale_bits) {
    std::array<float, D> rot = src;
    hadamard256(rot);
    float amax = 0;
    for (int d = 0; d < D; ++d) amax = std::max(amax, std::fabs(rot[d]));
    const float bounded = amax <= 0 ? 0.f : std::min(65504.f, std::max(0x1p-24f, amax / 7.f));
    scale_bits = (std::uint16_t)f32_to_f16_bits(bounded);
    const float s   = f16_bits_to_f32(scale_bits);
    const float inv = s > 0 ? 1.f / s : 0.f;
    codes.assign(D / 2, 0);
    for (int c = 0; c < D / 8; ++c) {
        float scaled[8], proj[8];
        for (int i = 0; i < 8; ++i) scaled[i] = rot[c * 8 + i] * inv;
        if (lattice) e8_project_host(scaled, proj);
        else for (int i = 0; i < 8; ++i) proj[i] = scaled[i];
        for (int i = 0; i < 8; ++i) {
            const int lo    = lattice ? -8 : -7;
            int code        = (int)std::nearbyint(proj[i]);
            code            = std::max(lo, std::min(7, code));
            const int dim   = c * 8 + i;
            const std::uint8_t nib = (std::uint8_t)code & 0xFu;
            if (dim & 1) codes[dim / 2] |= (std::uint8_t)(nib << 4);
            else         codes[dim / 2] |= nib;
        }
    }
}

int decode_nib(std::uint8_t byte, int high) {
    const unsigned n = high ? (byte >> 4) : (byte & 0xFu);
    return (int)(n ^ 8u) - 8;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

} // namespace

int main() {
    std::mt19937 rng(20260925u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<__nv_bfloat16> kh(D * KVH * T), vh(D * KVH * T);
    std::array<std::array<float, D>, KVH * T> k_src, v_src;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < KVH; ++h)
            for (int d = 0; d < D; ++d) {
                const int row = t * KVH + h;
                k_src[row][d] = dist(rng); v_src[row][d] = dist(rng);
            }
    // Mirror the attention suite's inject_codec_edges(): zero a 64-dim K block and a 64-dim V
    // block in head 0 / token 0, then plant one -1.0 K and one +1.0 V outlier in the last
    // head of the last token.
    for (int d = 0; d < 64; ++d) {
        k_src[0 * KVH + 0][d]      = 0.0f;
        v_src[0 * KVH + 0][64 + d] = 0.0f;
    }
    k_src[(T - 1) * KVH + (KVH - 1)][0] = -1.0f;
    v_src[(T - 1) * KVH + (KVH - 1)][0] =  1.0f;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < KVH; ++h)
            for (int d = 0; d < D; ++d) {
                const int row = t * KVH + h;
                kh[row * D + d] = __float2bfloat16_rn(k_src[row][d]);
                vh[row * D + d] = __float2bfloat16_rn(v_src[row][d]);
            }

    __nv_bfloat16 *dk, *dv; int *dpos;
    unsigned char *dkc, *dvc; __half *dks, *dvs; int *dtable;
    CK(cudaMalloc(&dk, D * KVH * T * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dv, D * KVH * T * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dpos, T * sizeof(int)));
    const std::size_t code_bytes  = (std::size_t)(D / 2) * PAGE * KVH;
    const std::size_t scale_bytes = (std::size_t)PAGE * KVH * sizeof(__half);
    CK(cudaMalloc(&dkc, code_bytes)); CK(cudaMalloc(&dvc, code_bytes));
    CK(cudaMalloc(&dks, scale_bytes)); CK(cudaMalloc(&dvs, scale_bytes));
    CK(cudaMalloc(&dtable, sizeof(int)));
    CK(cudaMemset(dkc, 0xAA, code_bytes)); CK(cudaMemset(dvc, 0xAA, code_bytes));
    CK(cudaMemset(dks, 0x55, scale_bytes)); CK(cudaMemset(dvs, 0x55, scale_bytes));
    const int page0 = 0;
    CK(cudaMemcpy(dtable, &page0, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, kh.data(), kh.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vh.data(), vh.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    int pos_host[T]; for (int t = 0; t < T; ++t) pos_host[t] = t;
    CK(cudaMemcpy(dpos, pos_host, sizeof(pos_host), cudaMemcpyHostToDevice));

    PagedKVLayerView cache;
    cache.k_pages       = Tensor(dkc, DType::U8, {D / 2, PAGE, KVH, 1});
    cache.v_pages       = Tensor(dvc, DType::U8, {D / 2, PAGE, KVH, 1});
    cache.k_scale_pages = Tensor(dks, DType::FP16, {1, PAGE, KVH, 1});
    cache.v_scale_pages = Tensor(dvs, DType::FP16, {1, PAGE, KVH, 1});
    cache.block_table   = Tensor(dtable, DType::I32, {1});
    cache.head_dim      = D; cache.num_kv_heads = KVH; cache.storage = KvCacheStorage::RK4V4E8;

    Tensor tk(dk, DType::BF16, {D, KVH, T});
    Tensor tv(dv, DType::BF16, {D, KVH, T});
    Tensor tp(dpos, DType::I32, {T});

    kv_cache_append(tk, tv, tp, cache, nullptr);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<unsigned char> kcb(code_bytes), vcb(code_bytes);
    CK(cudaMemcpy(kcb.data(), dkc, code_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vcb.data(), dvc, code_bytes, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < KVH; ++h) {
            const int row = t * KVH + h;
            const std::size_t base = (std::size_t)(h + KVH * t) * (D / 2);
            std::vector<std::uint8_t> kc, vc; std::uint16_t ksb = 0, vsb = 0;
            std::array<float, D> tmp;
            for (int d = 0; d < D; ++d) tmp[d] = __bfloat162float(kh[row * D + d]);
            oracle_codes(tmp, true, kc, ksb);
            for (int d = 0; d < D; ++d) tmp[d] = __bfloat162float(vh[row * D + d]);
            oracle_codes(tmp, false, vc, vsb);
            for (int i = 0; i < D / 2; ++i) {
                if (kcb[base + i] != kc[i]) {
                    std::printf("  K MISM t=%d h=%d byte=%d exp=0x%02x got=0x%02x"
                                " (dims %d,%d exp %d,%d got %d,%d)\n",
                                t, h, i, kc[i], kcb[base + i], 2 * i, 2 * i + 1,
                                decode_nib(kc[i], 0), decode_nib(kc[i], 1),
                                decode_nib(kcb[base + i], 0), decode_nib(kcb[base + i], 1));
                    ++bad; break;
                }
            }
            for (int i = 0; i < D / 2; ++i) {
                if (vcb[base + i] != vc[i]) {
                    std::printf("  V MISM t=%d h=%d byte=%d exp=0x%02x got=0x%02x"
                                " (dims %d,%d exp %d,%d got %d,%d)\n",
                                t, h, i, vc[i], vcb[base + i], 2 * i, 2 * i + 1,
                                decode_nib(vc[i], 0), decode_nib(vc[i], 1),
                                decode_nib(vcb[base + i], 0), decode_nib(vcb[base + i], 1));
                    ++bad; break;
                }
            }
        }
    std::printf(bad == 0 ? "PROBE2 PASS: T=%d fill matches host oracle\n"
                         : "PROBE2 FAIL: %d rows diverge\n", bad == 0 ? T : bad);
    return bad == 0 ? 0 : 1;
}
