#pragma once

#include "ninfer/ops/softmax_attention.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The rk4v4-e8 route keeps its own split-K policy: the split count bounds the partial reduction
// order of an FP32 softmax merge, so it belongs with the kernels that are qualified against it. The
// Blackwell tensor-core routes derive their splits from the shared SM-count partition; the E8 route
// is Ada-only SIMT/lattice code and reads the ladder below instead.
enum class E8KvFamily { SmallT, ChunkedSmallT, Prompt };

struct E8KvCausalPlan {
    E8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    std::int32_t splits;        // partial reduction lanes for one invocation
    std::int32_t chunk_tokens;  // query-token chunk width for the chunked family
};

// Splits for one invocation of `tokens` query columns over `envelope`, matching the retired
// cross-storage table evaluated for rk4v4-e8 (its per-storage tiers gated on BF16, INT8-G64,
// FP8-ROW and NVFP4 shapes, none of which the E8 route takes).
std::int32_t e8_kv_split_capacity(int q_heads, std::int32_t tokens,
                                  CausalAttentionExecutionEnvelope envelope, int batch_size = 1);

std::int32_t e8_kv_chunk_tokens(int q_heads);

E8KvCausalPlan make_e8_kv_causal_plan(int heads, int width, int batch,
                                      CausalAttentionExecutionEnvelope envelope,
                                      int multiprocessor_count);

std::size_t e8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                  CausalAttentionExecutionEnvelope envelope,
                                  int multiprocessor_count);

} // namespace ninfer::ops::detail
