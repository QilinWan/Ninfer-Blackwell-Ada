#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;

struct GroupedShape {
    int resident_ctas; // CTAs per SM guaranteed by the schedule's launch bound
    int key_shift;     // log2 of the schedule's key tile, the smallest split
};

template <class G>
GroupedShape grouped_shape(int tokens) {
    const auto of = []<int Tokens>() {
        using S = typename Int8KvGroupedInstance<G, Tokens>::Schedule;
        static_assert(S::kKeyRows == 32 || S::kKeyRows == 64);
        return GroupedShape{S::kMinBlocks, S::kKeyRows == 32 ? 5 : 6};
    };
    switch (tokens) {
    case 1: return of.template operator()<1>();
    case 2: return of.template operator()<2>();
    case 3: return of.template operator()<3>();
    case 4: return of.template operator()<4>();
    case 5: return of.template operator()<5>();
    case 6: return of.template operator()<6>();
    case 7: return of.template operator()<7>();
    default: return of.template operator()<8>();
    }
}

GroupedShape grouped_shape(int heads, int tokens) {
    if (heads == 24) return grouped_shape<CausalD256H24Kv4>(tokens);
    if (heads == 12) return grouped_shape<CausalD256H12Kv2>(tokens);
    return grouped_shape<CausalD256H16Kv2>(tokens);
}
} // namespace

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16 && heads != 12) || width < 1 || batch < 1 ||
        batch > 8 || (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("INT8 attention: invalid plan inputs");
    constexpr int grouped_limit = Int8KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? Int8KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Int8KvFamily::ParallelGrouped
                                                                     : Int8KvFamily::Tiled;
    const int tiles =
        family == Int8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    const std::int64_t sms      = multiprocessor_count;
    const auto budget =
        heads == 24 || width <= 4 ||
                causal_query_tiles_underfill_sms(independent_tiles, multiprocessor_count)
            ? 2 * sms
            : sms;
    CausalKvPartition partition{1, causal_partition_target(budget, independent_tiles)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, width, batch, envelope, partition};
}

std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            make_int8_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        if (plan.family == Int8KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
