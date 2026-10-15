// The per-rank forms of the DFlash2 drafter at tensor-parallel width 2, on one device, against the
// complete profiles they split:
//
//   1. rmsnorm_rope_head_block: q [128,16,W,B] / k [128,4,W,B] equal the 32/8 pair's heads
//      [16 r, 16 r + 16) / [4 r, 4 r + 4) bit for bit;
//   2. attn_input_proj_head_block: the Q8 [3072,5120] shard (a rank's Q|K|V rows of the
//      [6144,5120] parent) writes the parent's q/k/v rows bit for bit, at every route's widths;
//   3. sliding_window_attention_head_block: 16/4 heads over a D128/H4 ring equal the 32/8 output's
//      heads bit for bit, direct and split routes;
//   4. context_kv_materialize_head_block: [512,5120] key/value row blocks write H4 rings equal to
//      the parent's rings' heads bit for bit, decode blocks and a prefill chunk;
//   5. dynamic_grouped_conv_finish_add against an FP64 oracle (one BF16 rounding);
//   6. the Q8 linear half n5120_k2048 against an FP64 oracle.
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/tp2/attn_input_proj.h"
#include "ninfer/ops/tp2/context_kv_materialize.h"
#include "ninfer/ops/tp2/dynamic_grouped_conv.h"
#include "ninfer/ops/tp2/rmsnorm_rope.h"
#include "ninfer/ops/tp2/sliding_window_attention.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr int D = 128, Hidden = 5120;

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = f32_to_bf16(v[i]);
    return out;
}

// Copies heads [first, first + count) of a [D, heads, tokens] BF16/FP16 bit array.
std::vector<std::uint16_t> head_block(const std::vector<std::uint16_t>& all, int heads, int tokens,
                                      int first, int count) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(D) * count * tokens);
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < count; ++h)
            std::copy_n(all.begin() + (static_cast<std::size_t>(t) * heads + first + h) * D, D,
                        out.begin() + (static_cast<std::size_t>(t) * count + h) * D);
    return out;
}

// ---------------------------------------------------------------- 1. rmsnorm_rope
int run_rmsnorm_rope() {
    int failures = 0;
    for (const auto [W, B] : {std::pair{8, 1}, std::pair{8, 4}, std::pair{16, 2}}) {
        const int tokens = W * B;
        std::vector<float> q(static_cast<std::size_t>(D) * 32 * tokens), k(D * 8 * tokens),
            qn(D), kn(D);
        fill_uniform(q, 11U + W * B, -2.f, 2.f);
        fill_uniform(k, 13U + W * B, -2.f, 2.f);
        fill_uniform(qn, 17U, 0.5f, 1.5f);
        fill_uniform(kn, 19U, 0.5f, 1.5f);
        std::vector<int> positions(tokens);
        for (int i = 0; i < tokens; ++i) positions[i] = 1000 + 37 * i;
        const auto qb = to_bf16_bits(q), kb = to_bf16_bits(k);
        DeviceBuffer dq = to_device(qb), dk = to_device(kb), dqn = to_device_bf16(qn),
                     dkn = to_device_bf16(kn), dp = to_device(positions);
        Tensor tq(dq.p, DType::BF16, {D, 32, W, B}), tk(dk.p, DType::BF16, {D, 8, W, B});
        const Tensor tqn(dqn.p, DType::BF16, {D}), tkn(dkn.p, DType::BF16, {D}),
            tp(dp.p, DType::I32, {W, B});
        ops::rmsnorm_rope(tp, tqn, tkn, tq, tk, nullptr);
        cuda_synchronize();
        const auto q_all = from_device<std::uint16_t>(dq.p, qb.size());
        const auto k_all = from_device<std::uint16_t>(dk.p, kb.size());
        for (int r = 0; r < 2; ++r) {
            DeviceBuffer rq = to_device(head_block(qb, 32, tokens, 16 * r, 16));
            DeviceBuffer rk = to_device(head_block(kb, 8, tokens, 4 * r, 4));
            Tensor rtq(rq.p, DType::BF16, {D, 16, W, B}), rtk(rk.p, DType::BF16, {D, 4, W, B});
            ops::rmsnorm_rope_head_block(tp, tqn, tkn, rtq, rtk, nullptr);
            cuda_synchronize();
            const std::string label = "rmsnorm_rope W=" + std::to_string(W) +
                                      " B=" + std::to_string(B) + " rank " + std::to_string(r);
            failures += verify_exact((label + " q").c_str(),
                                     from_device<std::uint16_t>(rq.p, D * 16 * tokens),
                                     head_block(q_all, 32, tokens, 16 * r, 16));
            failures += verify_exact((label + " k").c_str(),
                                     from_device<std::uint16_t>(rk.p, D * 4 * tokens),
                                     head_block(k_all, 8, tokens, 4 * r, 4));
        }
    }
    return failures;
}

// Rows [begin, begin + count) of a RowSplit Q8 payload (codes then scales), appended to `out`'s
// code and scale planes at row `at`.
void copy_q8_rows(const qw::PackedWeight& parent, qw::PackedWeight& shard, int begin, int count,
                  int at) {
    const std::size_t code_row  = static_cast<std::size_t>(parent.weight.k);
    const std::size_t scale_row = static_cast<std::size_t>(parent.weight.k / 32) * 2;
    std::copy_n(parent.payload.data() + begin * code_row, count * code_row,
                shard.payload.data() + at * code_row);
    std::copy_n(parent.payload.data() + parent.scale_plane_offset + begin * scale_row,
                count * scale_row, shard.payload.data() + shard.scale_plane_offset + at * scale_row);
}

// ---------------------------------------------------------------- 2. attn_input_proj shard
int run_attn_input() {
    int failures                  = 0;
    const qw::PackedWeight parent = qw::make_patterned_weight(QType::Q8_G32_FP16, 6144, Hidden, 77U);
    DeviceBuffer dparent          = to_device(parent.payload);
    const Weight wparent          = parent.device_weight(dparent.p);
    std::array<qw::PackedWeight, 2> shard{
        qw::make_patterned_weight(QType::Q8_G32_FP16, 3072, Hidden, 1U),
        qw::make_patterned_weight(QType::Q8_G32_FP16, 3072, Hidden, 2U)};
    std::array<DeviceBuffer, 2> dshard;
    for (int r = 0; r < 2; ++r) {
        copy_q8_rows(parent, shard[r], 2048 * r, 2048, 0);
        copy_q8_rows(parent, shard[r], 4096 + 512 * r, 512, 2048);
        copy_q8_rows(parent, shard[r], 5120 + 512 * r, 512, 2560);
        dshard[r] = to_device(shard[r].payload);
    }
    for (const int T : {1, 7, 8, 14, 28, 32, 48, 56, 63, 64}) {
        std::vector<float> x(static_cast<std::size_t>(Hidden) * T);
        fill_uniform(x, 23U + T, -1.f, 1.f);
        DeviceBuffer dx = to_device_bf16(x);
        const Tensor tx(dx.p, DType::BF16, {Hidden, T});
        DeviceBuffer pq(4096 * T * 2), pk(1024 * T * 2), pv(1024 * T * 2);
        Tensor tq(pq.p, DType::BF16, {4096, T}), tk(pk.p, DType::BF16, {1024, T}),
            tv(pv.p, DType::BF16, {1024, T});
        ops::attn_input_proj(tx, wparent, tq, tk, tv, nullptr);
        cuda_synchronize();
        const auto q_all = from_device<std::uint16_t>(pq.p, 4096 * T);
        const auto k_all = from_device<std::uint16_t>(pk.p, 1024 * T);
        const auto v_all = from_device<std::uint16_t>(pv.p, 1024 * T);
        for (int r = 0; r < 2; ++r) {
            DeviceBuffer sq(2048 * T * 2), sk(512 * T * 2), sv(512 * T * 2);
            Tensor tsq(sq.p, DType::BF16, {2048, T}), tsk(sk.p, DType::BF16, {512, T}),
                tsv(sv.p, DType::BF16, {512, T});
            ops::attn_input_proj_head_block(tx, shard[r].device_weight(dshard[r].p), tsq, tsk,
                                            tsv, nullptr);
            cuda_synchronize();
            const std::string label =
                "attn_input_proj shard T=" + std::to_string(T) + " rank " + std::to_string(r);
            failures += verify_exact((label + " q").c_str(),
                                     from_device<std::uint16_t>(sq.p, 2048 * T),
                                     head_block(q_all, 32, T, 16 * r, 16));
            failures += verify_exact((label + " k").c_str(),
                                     from_device<std::uint16_t>(sk.p, 512 * T),
                                     head_block(k_all, 8, T, 4 * r, 4));
            failures += verify_exact((label + " v").c_str(),
                                     from_device<std::uint16_t>(sv.p, 512 * T),
                                     head_block(v_all, 8, T, 4 * r, 4));
        }
    }
    return failures;
}

// A [D, padded, heads, lanes] ring's heads [first, first + count).
std::vector<std::uint16_t> ring_block(const std::vector<std::uint16_t>& all, int padded, int heads,
                                      int lanes, int first, int count) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(D) * padded * count * lanes);
    const std::size_t head_elements = static_cast<std::size_t>(D) * padded;
    for (int lane = 0; lane < lanes; ++lane)
        for (int h = 0; h < count; ++h)
            std::copy_n(all.begin() + (static_cast<std::size_t>(lane) * heads + first + h) *
                                          head_elements,
                        head_elements,
                        out.begin() + (static_cast<std::size_t>(lane) * count + h) * head_elements);
    return out;
}

CyclicKVCacheLayerView ring_view(void* k, void* v, int window, int padded, int heads, int lanes) {
    return {.k               = Tensor(k, DType::BF16, {D, padded, heads, lanes}),
            .v               = Tensor(v, DType::FP16, {D, padded, heads, lanes}),
            .capacity        = static_cast<std::uint32_t>(window),
            .padded_capacity = static_cast<std::uint32_t>(padded),
            .num_kv_heads    = heads,
            .head_dim        = D,
            .lane_capacity   = lanes};
}

// ---------------------------------------------------------------- 3. sliding window attention
int run_sliding_window() {
    int failures             = 0;
    constexpr int Window     = 2048, Padded = 2056, Lanes = 4;
    constexpr float Scale    = 0.08838834764831844055f;
    std::vector<float> kf(static_cast<std::size_t>(D) * Padded * 8 * Lanes), vf(kf.size());
    fill_uniform(kf, 401U, -0.4f, 0.4f);
    fill_uniform(vf, 503U, -0.8f, 0.8f);
    const auto k_bits = to_bf16_bits(kf);
    std::vector<std::uint16_t> v_bits(vf.size());
    for (std::size_t i = 0; i < vf.size(); ++i) {
        const __half h = __float2half_rn(vf[i]);
        std::memcpy(&v_bits[i], &h, 2);
    }
    DeviceBuffer ck = to_device(k_bits), cv = to_device(v_bits);
    std::array<DeviceBuffer, 2> rk, rv;
    for (int r = 0; r < 2; ++r) {
        rk[r] = to_device(ring_block(k_bits, Padded, 8, Lanes, 4 * r, 4));
        rv[r] = to_device(ring_block(v_bits, Padded, 8, Lanes, 4 * r, 4));
    }
    // (W, B, context length): a direct route (short context) and split routes.
    for (const auto [W, B, L] : {std::tuple{8, 1, 40}, std::tuple{8, 1, 1500},
                                 std::tuple{8, 3, 3000}, std::tuple{16, 2, 700}}) {
        const int tokens = W * B;
        std::vector<float> q(static_cast<std::size_t>(D) * 32 * tokens), qk(D * 8 * tokens),
            qv(D * 8 * tokens);
        fill_uniform(q, 31U + L, -1.f, 1.f);
        fill_uniform(qk, 37U + L, -0.4f, 0.4f);
        fill_uniform(qv, 41U + L, -0.8f, 0.8f);
        const auto qb = to_bf16_bits(q), qkb = to_bf16_bits(qk), qvb = to_bf16_bits(qv);
        std::vector<int> positions(tokens), valid(B), lanes(B);
        for (int b = 0; b < B; ++b) {
            valid[b] = W - (b % 2);
            lanes[b] = (b + 1) % Lanes;
            for (int t = 0; t < W; ++t) positions[b * W + t] = L + 100 * b + t;
        }
        DeviceBuffer dq = to_device(qb), dqk = to_device(qkb), dqv = to_device(qvb),
                     dp = to_device(positions), dvalid = to_device(valid),
                     dlanes = to_device(lanes);
        const Tensor tp(dp.p, DType::I32, {W, B}), tvalid(dvalid.p, DType::I32, {B}),
            tlanes(dlanes.p, DType::I32, {B});
        const ops::SlidingWindowAttentionExecutionEnvelope envelope{
            0, static_cast<std::uint32_t>(L + 100 * B)};
        DeviceBuffer out(static_cast<std::size_t>(D) * 32 * tokens * 2);
        {
            WorkspaceArena ws(ops::sliding_window_attention_workspace_capacity_bytes(
                {D, 32, 8}, Window, envelope, W, W, B));
            Tensor tout(out.p, DType::BF16, {D, 32, W, B});
            ops::sliding_window_attention(
                Tensor(dq.p, DType::BF16, {D, 32, W, B}), Tensor(dqk.p, DType::BF16, {D, 8, W, B}),
                Tensor(dqv.p, DType::BF16, {D, 8, W, B}), tp, tvalid, tlanes, {D, 32, 8}, Window,
                Scale, ring_view(ck.p, cv.p, Window, Padded, 8, Lanes), envelope, ws, tout,
                nullptr);
            cuda_synchronize();
        }
        const auto out_all = from_device<std::uint16_t>(out.p, D * 32 * tokens);
        for (int r = 0; r < 2; ++r) {
            DeviceBuffer rq = to_device(head_block(qb, 32, tokens, 16 * r, 16));
            DeviceBuffer rqk = to_device(head_block(qkb, 8, tokens, 4 * r, 4));
            DeviceBuffer rqv = to_device(head_block(qvb, 8, tokens, 4 * r, 4));
            DeviceBuffer rout(static_cast<std::size_t>(D) * 16 * tokens * 2);
            WorkspaceArena ws(ops::sliding_window_attention_head_block_workspace_capacity_bytes(
                {D, 16, 4}, Window, envelope, W, W, B));
            Tensor tout(rout.p, DType::BF16, {D, 16, W, B});
            ops::sliding_window_attention_head_block(
                Tensor(rq.p, DType::BF16, {D, 16, W, B}), Tensor(rqk.p, DType::BF16, {D, 4, W, B}),
                Tensor(rqv.p, DType::BF16, {D, 4, W, B}), tp, tvalid, tlanes, {D, 16, 4}, Window,
                Scale, ring_view(rk[r].p, rv[r].p, Window, Padded, 4, Lanes), envelope, ws, tout,
                nullptr);
            cuda_synchronize();
            failures += verify_exact(("sliding window W=" + std::to_string(W) + " B=" +
                                      std::to_string(B) + " L=" + std::to_string(L) + " rank " +
                                      std::to_string(r))
                                         .c_str(),
                                     from_device<std::uint16_t>(rout.p, D * 16 * tokens),
                                     head_block(out_all, 32, tokens, 16 * r, 16));
        }
    }
    return failures;
}

// ---------------------------------------------------------------- 4. context K/V
int run_context_kv() {
    int failures         = 0;
    constexpr int Padded = 2056, Lanes = 4, Layers = ops::kContextKVMaterializeLayers;
    const qw::PatternedWeightOptions options{qw::RowSplitScalePattern::Tiny};
    std::vector<qw::PackedWeight> key, value;
    std::vector<DeviceBuffer> dkey, dvalue, dnorm;
    for (int l = 0; l < Layers; ++l) {
        key.push_back(qw::make_patterned_weight(QType::Q8_G32_FP16, 1024, Hidden, 0x310U + 2 * l,
                                                options));
        value.push_back(qw::make_patterned_weight(QType::Q8_G32_FP16, 1024, Hidden,
                                                  0x311U + 2 * l, options));
        dkey.push_back(to_device(key.back().payload));
        dvalue.push_back(to_device(value.back().payload));
        std::vector<float> norm(D);
        fill_uniform(norm, 71U + l, 0.6f, 1.4f);
        dnorm.push_back(to_device_bf16(norm));
    }
    // A rank's [512,5120] block as a view into the parent's payload: 512 contiguous code rows and
    // 512 contiguous scale rows.
    const auto block = [&](const qw::PackedWeight& w, const DeviceBuffer& d, int r) {
        qw::PackedWeight geometry =
            qw::make_patterned_weight(QType::Q8_G32_FP16, 512, Hidden, 1U, options);
        Weight view  = geometry.device_weight(d.p);
        view.qdata   = static_cast<const std::uint8_t*>(d.p) + std::size_t(512) * r * Hidden;
        view.scales  = static_cast<const std::uint8_t*>(d.p) + w.scale_plane_offset +
                      std::size_t(512) * r * (Hidden / 32) * 2;
        view.payload_bytes = w.payload.size();
        return view;
    };
    const std::size_t ring8 = static_cast<std::size_t>(D) * Padded * 8 * Lanes;
    for (const auto [W, B] : {std::pair{8, 1}, std::pair{8, 4}, std::pair{16, 2},
                              std::pair{300, 1}, std::pair{1024, 1}}) {
        const int tokens = W * B;
        std::vector<float> context(static_cast<std::size_t>(Hidden) * tokens);
        fill_uniform(context, 91U + tokens, -1.f, 1.f);
        DeviceBuffer dctx = to_device_bf16(context);
        std::vector<int> positions(tokens), counts(B), slots(B);
        for (int b = 0; b < B; ++b) {
            counts[b] = W - (b % 3);
            slots[b]  = (b * 3 + 1) % Lanes;
            for (int t = 0; t < W; ++t) positions[b * W + t] = 5000 + 1000 * b + t;
        }
        int min_count = W, max_count = 0;
        for (const int c : counts) min_count = std::min(min_count, c), max_count = std::max(max_count, c);
        DeviceBuffer dp = to_device(positions), dc = to_device(counts), ds = to_device(slots);
        const Tensor tctx(dctx.p, DType::BF16, {Hidden, W, B}), tp(dp.p, DType::I32, {W, B}),
            tc(dc.p, DType::I32, {B}), ts(ds.p, DType::I32, {B});
        const ops::ContextKVMaterializeExecutionEnvelope envelope{
            static_cast<std::uint32_t>(min_count), static_cast<std::uint32_t>(max_count)};
        std::vector<DeviceBuffer> pk, pv;
        std::array<ops::ContextKVMaterializeLayerView, Layers> parent_layers;
        for (int l = 0; l < Layers; ++l) {
            pk.emplace_back(ring8 * 2);
            pv.emplace_back(ring8 * 2);
            cuda_check(cudaMemset(pk.back().p, 0, ring8 * 2), "memset");
            cuda_check(cudaMemset(pv.back().p, 0, ring8 * 2), "memset");
            parent_layers[l] = {key[l].device_weight(dkey[l].p),
                                value[l].device_weight(dvalue[l].p),
                                Tensor(dnorm[l].p, DType::BF16, {D}),
                                ring_view(pk.back().p, pv.back().p, 2048, Padded, 8, Lanes)};
        }
        {
            WorkspaceArena ws(ops::context_kv_materialize_workspace_capacity_bytes(B, W, W));
            ops::context_kv_materialize(tctx, tp, tc, ts, parent_layers, envelope, ws, nullptr);
            cuda_synchronize();
        }
        for (int r = 0; r < 2; ++r) {
            std::vector<DeviceBuffer> rk, rv;
            std::array<ops::ContextKVMaterializeLayerView, Layers> rank_layers;
            for (int l = 0; l < Layers; ++l) {
                rk.emplace_back(ring8);
                rv.emplace_back(ring8);
                cuda_check(cudaMemset(rk.back().p, 0, ring8), "memset");
                cuda_check(cudaMemset(rv.back().p, 0, ring8), "memset");
                rank_layers[l] = {block(key[l], dkey[l], r), block(value[l], dvalue[l], r),
                                  Tensor(dnorm[l].p, DType::BF16, {D}),
                                  ring_view(rk.back().p, rv.back().p, 2048, Padded, 4, Lanes)};
            }
            WorkspaceArena ws(ops::context_kv_materialize_head_block_workspace_capacity_bytes(B, W, W));
            ops::context_kv_materialize_head_block(tctx, tp, tc, ts, rank_layers, envelope, ws,
                                                   nullptr);
            cuda_synchronize();
            for (int l = 0; l < Layers; ++l) {
                const std::string label = "context kv W=" + std::to_string(W) + " B=" +
                                          std::to_string(B) + " layer " + std::to_string(l) +
                                          " rank " + std::to_string(r);
                failures += verify_exact(
                    (label + " K").c_str(), from_device<std::uint16_t>(rk[l].p, ring8 / 2),
                    ring_block(from_device<std::uint16_t>(pk[l].p, ring8), Padded, 8, Lanes,
                               4 * r, 4));
                failures += verify_exact(
                    (label + " V").c_str(), from_device<std::uint16_t>(rv[l].p, ring8 / 2),
                    ring_block(from_device<std::uint16_t>(pv[l].p, ring8), Padded, 8, Lanes,
                               4 * r, 4));
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------- 5. conv finish
int run_finish() {
    constexpr int W = 8, B = 3, H = 5120, G = 320;
    std::vector<float> z(static_cast<std::size_t>(H) * W * B), base(H * 4),
        delta(static_cast<std::size_t>(G) * 2 * W * B), residual(z.size());
    fill_uniform(z, 5U, -2.f, 2.f);
    fill_uniform(base, 6U, -0.5f, 0.5f);
    fill_uniform(delta, 7U, -0.5f, 0.5f);
    fill_uniform(residual, 8U, -4.f, 4.f);
    for (auto* v : {&z, &base, &delta, &residual}) round_to_bf16(*v);
    DeviceBuffer dz = to_device_bf16(z), db = to_device_bf16(base), dd = to_device_bf16(delta),
                 dr = to_device_bf16(residual);
    Tensor tr(dr.p, DType::BF16, {H, W, B});
    ops::dynamic_grouped_conv_finish_add(Tensor(dz.p, DType::BF16, {H, W, B}),
                                         Tensor(db.p, DType::BF16, {H, 2, 2}),
                                         Tensor(dd.p, DType::BF16, {G, 2, W, B}), tr, nullptr);
    cuda_synchronize();
    const auto got = from_device_bf16(dr, z.size());
    int bad        = 0;
    for (int col = 0; col < W * B; ++col)
        for (int h = 0; h < H; ++h) {
            const std::size_t i = static_cast<std::size_t>(col) * H + h;
            const int di        = col * 2 * G + h / 16;
            double value        = double(residual[i]) + (double(base[2 * H + h]) + delta[di]) * z[i];
            if (col % W != 0) value += (double(base[3 * H + h]) + delta[di + G]) * z[i - H];
            const double tol = std::max(1e-6, std::abs(value) / 128.0);
            if (std::abs(got[i] - value) > tol) ++bad;
        }
    if (bad) std::cerr << "conv finish: " << bad << " values beyond one BF16 rounding\n";
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------- 6. Q8 linear n5120_k2048
int run_linear_half() {
    int failures             = 0;
    const qw::PackedWeight w = qw::make_patterned_weight(QType::Q8_G32_FP16, 5120, 2048, 99U);
    DeviceBuffer dw          = to_device(w.payload);
    const Weight weight      = w.device_weight(dw.p);
    for (const int T : {1, 8, 16, 32, 56, 64, 100}) {
        std::vector<float> x(static_cast<std::size_t>(2048) * T);
        fill_uniform(x, 3U + T, -1.f, 1.f);
        round_to_bf16(x);
        DeviceBuffer dx = to_device_bf16(x), dout(static_cast<std::size_t>(5120) * T * 2);
        Tensor tout(dout.p, DType::BF16, {5120, T});
        WorkspaceArena ws(ops::linear_workspace_capacity_bytes(QType::Q8_G32_FP16, 5120, 2048,
                                                              ops::LinearPolicy::A16Only, T, T) +
                          256);
        ops::linear(Tensor(dx.p, DType::BF16, {2048, T}), weight, tout,
                    ops::LinearPolicy::A16Only, ws, nullptr);
        cuda_synchronize();
        const auto got = from_device_bf16(dout, static_cast<std::size_t>(5120) * T);
        int bad        = 0;
        for (int t = 0; t < T; t += std::max(1, T / 7))
            for (int n = 0; n < 5120; n += 37) {
                double ref = 0.0, mag = 0.0;
                for (int k = 0; k < 2048; ++k) {
                    const double term = qw::logical_weight_fp64(w, n, k) * x[std::size_t(t) * 2048 + k];
                    ref += term;
                    mag += std::abs(term);
                }
                if (std::abs(got[std::size_t(t) * 5120 + n] - ref) > 1e-2 * mag / 32 + std::abs(ref) / 128)
                    ++bad;
            }
        if (bad) {
            std::cerr << "linear n5120_k2048 T=" << T << ": " << bad << " values off\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        const auto stage = [&](const char* name, int f) {
            std::cout << (f ? "FAIL " : "OK ") << name << '\n';
            failures += f;
        };
        stage("rmsnorm_rope head block", run_rmsnorm_rope());
        stage("attn_input_proj head block", run_attn_input());
        stage("sliding_window_attention head block", run_sliding_window());
        stage("context_kv_materialize head block", run_context_kv());
        stage("dynamic_grouped_conv_finish_add", run_finish());
        stage("linear q8 n5120_k2048", run_linear_half());
    } catch (const std::exception& error) {
        std::cerr << "dflash2 head blocks: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " dflash2 head blocks\n";
    return failures ? 1 : 0;
}
