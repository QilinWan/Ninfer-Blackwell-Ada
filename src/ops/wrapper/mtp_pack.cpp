#include "ninfer/ops/mtp_pack.h"
#include "ops/launcher/mtp_pack.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

// The fused query|key|gate|value projection of the whole model: 24 query/gate heads and 4 KV heads
// of 256 rows each. A two-device rank owns half of each head group.
constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKvHeads    = 4;

constexpr std::int32_t attn_in_rows(std::int32_t query_heads, std::int32_t kv_heads) {
    return 2 * (query_heads + kv_heads) * kHeadDim;
}

static_assert(attn_in_rows(kQueryHeads, kKvHeads) == 14336);
static_assert(attn_in_rows(kQueryHeads / 2, kKvHeads / 2) == 7168);

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, const char* op,
                   const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

} // namespace

void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_pack_fc_input";
    require_bf16_contiguous_nonnull(embedding_norm, op, "embedding_norm");
    require_bf16_contiguous_nonnull(hidden_norm, op, "hidden_norm");
    require_bf16_contiguous_nonnull(out, op, "out");
    const std::int32_t rows   = embedding_norm.ne[0];
    const std::int32_t tokens = embedding_norm.ne[1];
    if (rows <= 0) { throw std::invalid_argument("mtp_pack_fc_input: D must be positive"); }
    if (tokens <= 0) { throw std::invalid_argument("mtp_pack_fc_input: T must be positive"); }
    require_shape(embedding_norm, rows, tokens, op, "embedding_norm");
    require_shape(hidden_norm, rows, tokens, op, "hidden_norm");
    require_shape(out, 2 * rows, tokens, op, "out");

    detail::mtp_pack_fc_input_launch(embedding_norm, hidden_norm, out, stream);
}

void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_split_attn_in";
    require_bf16_contiguous_nonnull(attn_in, op, "attn_in");
    require_bf16_contiguous_nonnull(q, op, "q");
    require_bf16_contiguous_nonnull(k, op, "k");
    require_bf16_contiguous_nonnull(gate, op, "gate");
    require_bf16_contiguous_nonnull(v, op, "v");
    const std::int32_t tokens = attn_in.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("mtp_split_attn_in: T must be positive"); }
    // The whole projection [14336,T] with 24 query/gate and 4 KV heads, or one rank's two-device
    // shard [7168,T] with 12 and 2.
    const bool shard = attn_in.ne[0] == attn_in_rows(kQueryHeads / 2, kKvHeads / 2);
    if (attn_in.ne[0] != attn_in_rows(kQueryHeads, kKvHeads) && !shard) {
        throw std::invalid_argument("mtp_split_attn_in: unregistered attn_in row geometry");
    }
    const std::int32_t q_heads  = shard ? kQueryHeads / 2 : kQueryHeads;
    const std::int32_t kv_heads = shard ? kKvHeads / 2 : kKvHeads;
    require_shape(attn_in, attn_in.ne[0], tokens, op, "attn_in");
    if (q.ne[0] != kHeadDim || q.ne[1] != q_heads || q.ne[2] != tokens || q.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for q");
    }
    if (k.ne[0] != kHeadDim || k.ne[1] != kv_heads || k.ne[2] != tokens || k.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for k");
    }
    if (gate.ne[0] != kHeadDim || gate.ne[1] != q_heads || gate.ne[2] != tokens ||
        gate.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for gate");
    }
    if (v.ne[0] != kHeadDim || v.ne[1] != kv_heads || v.ne[2] != tokens || v.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for v");
    }

    detail::mtp_split_attn_in_launch(attn_in, q, k, gate, v, stream);
}

} // namespace ninfer::ops
