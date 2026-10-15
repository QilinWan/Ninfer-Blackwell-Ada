#include "ops/softmax_attention/dense/causal_cache/e8/plan.h"

#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/common/causal_operands.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kMaximumVerifyTokens        = 16;
constexpr std::uint32_t kTwoChunkPromptVisibleKeys = 512;
constexpr std::uint32_t kThreeChunkPromptVisibleKeys = 1024;

// The geometry ladder that selected the split count for a KV window, kept verbatim from the table
// the retired shared small-T launcher applied to rk4v4-e8. `SplitScale` and `MaximumSplits` are the
// per-geometry constants of the two registered head shapes (24Q/4KV scale 1, 16Q/2KV scale 2),
// inlined here because the route-private table is the only remaining reader.
template <int SplitScale, int MaximumSplits>
std::int32_t e8_split_upper_bound(std::int32_t window) {
    static_assert(SplitScale > 0);
    if (window <= 0) { return MaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * SplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / SplitScale);
    if (window > 4096) { include_tier(8198, 128 / SplitScale); }
    if (window > 8198) { include_tier(16390, 256 / SplitScale); }
    if (window > 16390) { include_tier(window, 480 / SplitScale); }

    return (splits < MaximumSplits) ? splits : MaximumSplits;
}

// The interval maximum of the ladder: the policy is monotonic inside each finite tier, so the
// segment ends plus both envelope ends give the exact capacity a launch must reserve space for.
template <int SplitScale, int MaximumSplits>
std::int32_t e8_launch_capacity(CausalAttentionExecutionEnvelope envelope) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits = e8_split_upper_bound<SplitScale, MaximumSplits>(
            static_cast<std::int32_t>(window));
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

} // namespace

std::int32_t e8_kv_split_capacity(int q_heads, std::int32_t tokens,
                                  CausalAttentionExecutionEnvelope envelope, int batch_size) {
    if (tokens < 1 || tokens > (q_heads == 24 ? 8 : 6) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("rk4v4-e8 attention: split capacity profile is invalid");
    }
    (void)paged_kv_storage_layout(KvCacheStorage::RK4V4E8, kCausalHeadDim);
    if (q_heads == CausalD256H24Kv4::QHeads) {
        const int capacity = e8_launch_capacity<1, 85>(envelope);
        if (batch_size > 1) {
            // Keep complete grids within one wave of the largest supported device: 160 CTAs at
            // 4 KV heads, with room for the indivisible 4*B group for B=3/5/6/7. A split stages
            // at most 64 physical-page IDs, and two 64-key pages are left to key-tile rounding and
            // page alignment at the resource limit.
            constexpr int kTargetCtas = 160;
            const int grid_limit      = div_up(kTargetCtas, 4 * batch_size);
            const int page_limit = div_up(static_cast<int>(envelope.max_visible_keys), 3968);
            return std::min(capacity, std::max({4, grid_limit, page_limit}));
        }
        return capacity;
    }
    if (q_heads == CausalD256H16Kv2::QHeads) { return e8_launch_capacity<2, 170>(envelope); }
    throw std::invalid_argument("rk4v4-e8 attention: unsupported head geometry");
}

std::int32_t e8_kv_chunk_tokens(int q_heads) {
    // The retired chunk rule balanced narrow BF16 and INT8-G64 chunks, neither of which is an E8
    // shape, so E8 keeps the two remaining terms: a 6-token chunk on the 16-head geometry.
    return q_heads == 16 ? 6 : 8;
}

E8KvCausalPlan make_e8_kv_causal_plan(int heads, int width, int batch,
                                      CausalAttentionExecutionEnvelope envelope,
                                      int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        batch > 8 || (batch > 1 && width > kMaximumVerifyTokens) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument("rk4v4-e8 attention: invalid plan inputs");
    }
    E8KvFamily family;
    if (heads == 24 && width <= kMaximumVerifyTokens) {
        if (batch == 1) {
            // E8 reads K as exact int8 codes through the same s8 QK machinery as the int8 route, so
            // it inherits the int8 prompt limit.
            const std::uint32_t prompt_limit = width <= 8 ? 0U : 256U;
            if (envelope.max_visible_keys <= prompt_limit)
                family = E8KvFamily::Prompt;
            else
                family = width <= 8 ? E8KvFamily::SmallT : E8KvFamily::ChunkedSmallT;
        } else {
            family = E8KvFamily::ChunkedSmallT;
        }
    } else if (width <= 6) {
        family = E8KvFamily::SmallT;
    } else if (batch > 1) {
        family = E8KvFamily::ChunkedSmallT;
    } else if (heads == 16 && width <= kMaximumVerifyTokens &&
               envelope.max_visible_keys >
                   (width <= 12 ? kTwoChunkPromptVisibleKeys : kThreeChunkPromptVisibleKeys)) {
        family = E8KvFamily::ChunkedSmallT;
    } else {
        family = E8KvFamily::Prompt;
    }
    const std::int32_t chunk = e8_kv_chunk_tokens(heads);
    const std::int32_t width_for_splits =
        family == E8KvFamily::ChunkedSmallT ? std::min(chunk, width) : width;
    return {family,
            heads,
            width,
            batch,
            envelope,
            e8_kv_split_capacity(heads, width_for_splits, envelope, batch),
            chunk};
}

std::size_t e8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                  CausalAttentionExecutionEnvelope envelope,
                                  int multiprocessor_count) {
    const auto chunk_bytes = [&](std::int32_t width, std::int32_t splits) {
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        return layout.peak_bytes(1);
    };
    std::size_t maximum = 0;
    for (int width = min_width; width <= max_width; ++width) {
        const auto plan = make_e8_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        if (plan.family == E8KvFamily::Prompt) { continue; }
        if (plan.family == E8KvFamily::SmallT) {
            maximum = std::max(maximum, chunk_bytes(width, plan.splits));
            continue;
        }
        for (int begin = 0; begin < width; begin += plan.chunk_tokens) {
            const int count = std::min(plan.chunk_tokens, width - begin);
            // One chunk reserves its own partial scope, so the peak is a single chunk's layout.
            maximum = std::max(maximum,
                               chunk_bytes(count, e8_kv_split_capacity(heads, count, envelope,
                                                                       batch)));
        }
    }
    return maximum;
}

} // namespace ninfer::ops::detail
