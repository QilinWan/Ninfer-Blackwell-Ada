#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kMtpAttnRows = 14336;
inline constexpr int kMtpQRows    = 6144;
inline constexpr int kMtpKvRows   = 1024;

// One rank's shard of the same projection under two-device tensor parallelism: half of each
// Q | K | Gate | V section, in the parent's order.
inline constexpr int kMtpAttnRowsTp2 = 7168;
inline constexpr int kMtpQRowsTp2    = 3072;
inline constexpr int kMtpKvRowsTp2   = 512;
static_assert(kMtpAttnRows == 2 * kMtpQRows + 2 * kMtpKvRows);
static_assert(kMtpAttnRowsTp2 == 2 * kMtpQRowsTp2 + 2 * kMtpKvRowsTp2);

__global__ void mtp_pack_fc_input_kernel(const __nv_bfloat16* embedding_norm,
                                         const __nv_bfloat16* hidden_norm, __nv_bfloat16* out,
                                         std::int32_t rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows + row;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    out[out_base + row]         = embedding_norm[in_idx];
    out[out_base + rows + row]  = hidden_norm[in_idx];
}

template <int AttnRows, int QRows, int KvRows>
__global__ void mtp_split_attn_in_kernel(const __nv_bfloat16* attn_in, __nv_bfloat16* q,
                                         __nv_bfloat16* k, __nv_bfloat16* gate, __nv_bfloat16* v,
                                         std::int32_t tokens) {
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(AttnRows) * tokens;
    if (idx >= n) { return; }

    const int row             = static_cast<int>(idx % AttnRows);
    const int token           = static_cast<int>(idx / AttnRows);
    const __nv_bfloat16 value = attn_in[idx];

    if (row < QRows) {
        q[static_cast<std::int64_t>(token) * QRows + row] = value;
        return;
    }
    if (row < QRows + KvRows) {
        const int local                                      = row - QRows;
        k[static_cast<std::int64_t>(token) * KvRows + local] = value;
        return;
    }
    if (row < QRows + KvRows + QRows) {
        const int local                                        = row - QRows - KvRows;
        gate[static_cast<std::int64_t>(token) * QRows + local] = value;
        return;
    }

    const int local                                      = row - QRows - KvRows - QRows;
    v[static_cast<std::int64_t>(token) * KvRows + local] = value;
}

} // namespace ninfer::ops
