// Non-RDC compilation is required for the producer/consumer register redistribution.
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                              cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_nvfp4_kv_tiled_mma<CausalD256H24Kv4, Nvfp4KvTiledInstance>(p, cache, stream);
    else if (p.query_heads == 16)
        launch_nvfp4_kv_tiled_mma<CausalD256H16Kv2, Nvfp4KvTiledInstance>(p, cache, stream);
    else
        throw std::invalid_argument("NVFP4 attention: unsupported head geometry");
}
} // namespace ninfer::ops::detail
