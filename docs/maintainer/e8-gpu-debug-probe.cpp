// E8 fill probe: standalone append of one T=1 key (kv_heads=4), then decode the device
// cache planes and compare decoded K/V values + scales against a host oracle
// (D256 Sylvester rotation, per-token absmax/7 FP16 scale, E8 projection / plain rint,
// two codes per byte). Prints the first divergences with device vs oracle bytes.
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

constexpr int D = 256;
constexpr int KVH = 4;
constexpr int T = 1;
constexpr int PAGE = 64;

// ---- host D256 Sylvester rotation (natural order, same matrix as the device warp) ----
void hadamard256(std::array<float, D>& v) {
    for (int stride = 1; stride < D; stride *= 2)
        for (int base = 0; base < D; base += 2 * stride)
            for (int off = 0; off < stride; ++off) {
                const float lo = v[base + off], hi = v[base + off + stride];
                v[base + off]         = lo + hi;
                v[base + off + stride] = lo - hi;
            }
    for (float& x : v) x *= 0x1p-4f;
}

// ---- host E8 nearest lattice projection (D8 / D8+0.5 with parity correction) ----
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
    return float(*reinterpret_cast<std::uint16_t*>(&h));
}
float f16_bits_to_f32(float bits) {
    std::uint16_t b = (std::uint16_t)bits;
    return __half2float(*reinterpret_cast<const __half*>(&b));
}

// Oracle decode target for one head row: returns rotated-domain decoded values and scale bits.
void oracle_row(const std::array<float, D>& src, bool lattice,
                std::array<float, D>& decoded, std::uint16_t& scale_bits) {
    std::array<float, D> rot = src;
    float amax = 0;
    for (int d = 0; d < D; ++d) amax = std::max(amax, std::fabs(rot[d]));
    hadamard256(rot);
    if (amax > 0) { amax = 0; for (int d = 0; d < D; ++d) amax = std::max(amax, std::fabs(rot[d])); }
    const float bounded = amax <= 0 ? 0.f : std::min(65504.f, std::max(0x1p-24f, amax / 7.f));
    scale_bits = (std::uint16_t)f32_to_f16_bits(bounded);
    const float s = f16_bits_to_f32(scale_bits);
    const float inv = s > 0 ? 1.f / s : 0.f;
    for (int c = 0; c < D / 8; ++c) {
        float scaled[8], proj[8];
        for (int i = 0; i < 8; ++i) scaled[i] = rot[c * 8 + i] * inv;
        if (lattice) e8_project_host(scaled, proj);
        else for (int i = 0; i < 8; ++i) proj[i] = scaled[i];
        for (int i = 0; i < 8; ++i) {
            int lo = lattice ? -8 : -7;
            int code = (int)std::nearbyint(proj[i]);
            code = std::max(lo, std::min(7, code));
            decoded[c * 8 + i] = code * s;
        }
    }
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

} // namespace

int main() {
    // ---- host data: one key, kv4 ----
    std::mt19937 rng(20260925u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<__nv_bfloat16> kh(D * KVH), vh(D * KVH);
    std::array<std::array<float, D>, KVH> k_src, v_src;
    for (int h = 0; h < KVH; ++h)
        for (int d = 0; d < D; ++d) {
            k_src[h][d] = dist(rng); v_src[h][d] = dist(rng);
            kh[h * D + d] = __float2bfloat16_rn(k_src[h][d]);
            vh[h * D + d] = __float2bfloat16_rn(v_src[h][d]);
        }

    // ---- device buffers ----
    __nv_bfloat16 *dk, *dv; int *dpos;
    unsigned char *dkc, *dvc; __half *dks, *dvs; int *dtable;
    CK(cudaMalloc(&dk, D * KVH * T * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dv, D * KVH * T * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dpos, T * sizeof(int)));
    const std::size_t code_bytes = (std::size_t)(D / 2) * PAGE * KVH;   // one page
    const std::size_t scale_bytes = (std::size_t)1 * PAGE * KVH * sizeof(__half);
    CK(cudaMalloc(&dkc, code_bytes)); CK(cudaMalloc(&dvc, code_bytes));
    CK(cudaMalloc(&dks, scale_bytes)); CK(cudaMalloc(&dvs, scale_bytes));
    CK(cudaMalloc(&dtable, sizeof(int)));
    CK(cudaMemset(dkc, 0xAA, code_bytes)); CK(cudaMemset(dvc, 0xAA, code_bytes));
    CK(cudaMemset(dks, 0x55, scale_bytes)); CK(cudaMemset(dvs, 0x55, scale_bytes));
    const int page0 = 0;
    CK(cudaMemcpy(dtable, &page0, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, kh.data(), kh.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vh.data(), vh.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    const int pos0 = 0;
    CK(cudaMemcpy(dpos, &pos0, sizeof(int), cudaMemcpyHostToDevice));

    // ---- view: E8 planes ----
    PagedKVLayerView cache;
    cache.k_pages = Tensor(dkc, DType::U8, {D / 2, PAGE, KVH, 1});
    cache.v_pages = Tensor(dvc, DType::U8, {D / 2, PAGE, KVH, 1});
    cache.k_scale_pages = Tensor(dks, DType::FP16, {1, PAGE, KVH, 1});
    cache.v_scale_pages = Tensor(dvs, DType::FP16, {1, PAGE, KVH, 1});
    cache.block_table = Tensor(dtable, DType::I32, {1});
    cache.head_dim = D; cache.num_kv_heads = KVH; cache.storage = KvCacheStorage::RK4V4E8;

    Tensor tk(dk, DType::BF16, {D, KVH, T});
    Tensor tv(dv, DType::BF16, {D, KVH, T});
    Tensor tp(dpos, DType::I32, {T});

    // ---- the standalone E8 append ----
    kv_cache_append(tk, tv, tp, cache, nullptr);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // ---- read back ----
    std::vector<unsigned char> kcb(code_bytes), vcb(code_bytes);
    std::vector<unsigned short> ksb(scale_bytes / 2), vsb(scale_bytes / 2);
    CK(cudaMemcpy(kcb.data(), dkc, code_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vcb.data(), dvc, code_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ksb.data(), dks, scale_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vsb.data(), dvs, scale_bytes, cudaMemcpyDeviceToHost));

    // ---- compare per head ----
    int fails = 0;
    for (int h = 0; h < KVH; ++h) {
        auto kdec_host = [](const unsigned char* b) {  // 128 bytes -> 256 codes
            std::array<int, D> c{};
            for (int i = 0; i < D / 2; ++i) {
                c[2 * i]     = int((b[i] & 0xF) ^ 8) - 8;
                c[2 * i + 1] = int((b[i] >> 4) ^ 8) - 8;
            }
            return c;
        };
        std::array<float, D> kdev{}, vdev{};
        auto kc = kdec_host(&kcb[(h * PAGE) * (D / 2)]);
        auto vc = kdec_host(&vcb[(h * PAGE) * (D / 2)]);
        const float ks = f16_bits_to_f32(ksb[h * PAGE]);
        const float vs = f16_bits_to_f32(vsb[h * PAGE]);
        std::array<float, D> kor{}, vor{};
        std::uint16_t ksb_or = 0, vsb_or = 0;
        { std::array<float, D> tmp; for (int d = 0; d < D; ++d) tmp[d] = k_src[h][d]; oracle_row(tmp, true, kor, ksb_or); }
        { std::array<float, D> tmp; for (int d = 0; d < D; ++d) tmp[d] = v_src[h][d]; oracle_row(tmp, false, vor, vsb_or); }
        for (int d = 0; d < D; ++d) { kdev[d] = float(kc[d]) * ks; vdev[d] = float(vc[d]) * vs; }

        const bool k_scale_ok = (ksb[h * PAGE] == ksb_or);
        const bool v_scale_ok = (vsb[h * PAGE] == vsb_or);
        double kerr = 0, verr = 0; int kbad = 0, vbad = 0;
        for (int d = 0; d < D; ++d) {
            const double e = std::fabs(kdev[d] - kor[d]); if (e > kerr) kerr = e; if (e > 0.6 * ks + 1e-6) ++kbad;
            const double e2 = std::fabs(vdev[d] - vor[d]); if (e2 > verr) verr = e2; if (e2 > 0.6 * vs + 1e-6) ++vbad;
        }
        std::printf("head %d: k_scale dev=%04x oracle=%04x %s | v_scale dev=%04x oracle=%04x %s\n",
                    h, ksb[h * PAGE], ksb_or, k_scale_ok ? "OK" : "MISMATCH",
                    vsb[h * PAGE], vsb_or, v_scale_ok ? "OK" : "MISMATCH");
        std::printf("        K: max|dev-oracle|=%.5f (scale=%.5f) bad_dims=%d | V: max=%.5f (scale=%.5f) bad_dims=%d\n",
                    kerr, ks, kbad, verr, vs, vbad);
        if (!k_scale_ok || !v_scale_ok || kbad || vbad) {
            ++fails;
            for (int d = 0; d < 32 && (kbad || vbad); ++d)
                if (std::fabs(kdev[d] - kor[d]) > 0.6 * ks + 1e-6 || std::fabs(vdev[d] - vor[d]) > 0.6 * vs + 1e-6) {
                    std::printf("        dim %2d: K dev=%.4f oracle=%.4f | V dev=%.4f oracle=%.4f | k_code dev=%d v_code dev=%d\n",
                                d, kdev[d], kor[d], vdev[d], vor[d], kc[d], vc[d]);
                }
        }
    }
    std::printf(fails == 0 ? "PROBE PASS: E8 fill matches host oracle\n"
                           : "PROBE FAIL: E8 fill diverges from oracle\n");
    return fails == 0 ? 0 : 1;
}
