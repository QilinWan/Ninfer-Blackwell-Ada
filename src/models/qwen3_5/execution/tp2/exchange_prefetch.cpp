#include "models/qwen3_5/execution/tp2/exchange_prefetch.h"

#include "core/device_scope.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

namespace {

struct RowGeometry {
    std::size_t code_stride  = 0; // bytes per row at qdata
    std::size_t scale_stride = 0; // bytes per row at scales (0: none prefetched)
    std::int64_t scale_tile  = 1; // rows per scale tile
};

RowGeometry row_geometry(const Weight& w) noexcept {
    const auto k = static_cast<std::size_t>(w.k);
    if (w.qdata == nullptr || w.n <= 0 || w.k <= 0) { return {}; }
    if (w.qtype == QType::NVFP4 && w.layout == QuantLayout::BlockScaleK16M128x4) {
        return {k / 2, w.scales != nullptr ? k / 16 : 0, 128};
    }
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16 && w.layout == QuantLayout::RowScale) {
        return {k, 0, 1};
    }
    if (w.qtype == QType::BF16) { return {2 * k, 0, 1}; }
    return {};
}

// [data, data + bytes) clipped to the weight's payload when it records one.
void append_clipped(const Weight& w, const void* data, std::size_t bytes, ExchangePrefetch& out) {
    auto begin = reinterpret_cast<std::uintptr_t>(data);
    auto end   = begin + bytes;
    if (w.payload != nullptr && w.payload_bytes != 0) {
        const auto lo = reinterpret_cast<std::uintptr_t>(w.payload);
        const auto hi = lo + w.payload_bytes;
        begin         = std::max(begin, lo);
        end           = std::min(end, hi);
    }
    if (end > begin) { out.push_back({reinterpret_cast<const void*>(begin), end - begin}); }
}

// NINFER_TP_PREFETCH=0 turns the hints off (the forward stages nothing).
bool prefetch_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TP_PREFETCH");
        return value == nullptr || std::string_view(value) != "0";
    }();
    return enabled;
}

} // namespace

std::size_t prefetch_row_bytes(const Weight& w) noexcept {
    const RowGeometry g = row_geometry(w);
    return g.code_stride + g.scale_stride;
}

void append_prefetch_rows(const Weight& w, std::int64_t row, std::int64_t rows,
                          ExchangePrefetch& out) {
    const RowGeometry g = row_geometry(w);
    if (g.code_stride == 0) { return; }
    const std::int64_t first = std::clamp<std::int64_t>(row, 0, w.n);
    const std::int64_t last  = std::clamp<std::int64_t>(row + rows, first, w.n);
    if (last == first) { return; }
    const auto* codes = static_cast<const std::uint8_t*>(w.qdata);
    append_clipped(w, codes + static_cast<std::size_t>(first) * g.code_stride,
                   static_cast<std::size_t>(last - first) * g.code_stride, out);
    if (g.scale_stride != 0) {
        const std::int64_t tile_first = first / g.scale_tile * g.scale_tile;
        const std::int64_t tile_last =
            std::min<std::int64_t>((last + g.scale_tile - 1) / g.scale_tile * g.scale_tile, w.n);
        const auto* scales = static_cast<const std::uint8_t*>(w.scales);
        append_clipped(w, scales + static_cast<std::size_t>(tile_first) * g.scale_stride,
                       static_cast<std::size_t>(tile_last - tile_first) * g.scale_stride, out);
    }
}

void append_prefetch_prefix(const Weight& w, std::size_t bytes, ExchangePrefetch& out) {
    const std::size_t row_bytes = prefetch_row_bytes(w);
    if (row_bytes == 0) { return; }
    append_prefetch_rows(w, 0, static_cast<std::int64_t>(bytes / row_bytes), out);
}

void append_prefetch_paired_prefix(const Weight& w, std::size_t bytes, ExchangePrefetch& out) {
    const std::size_t row_bytes = prefetch_row_bytes(w);
    if (row_bytes == 0 || w.n % 2 != 0) { return; }
    const auto rows         = static_cast<std::int64_t>(bytes / (2 * row_bytes));
    const std::int64_t half = w.n / 2;
    append_prefetch_rows(w, 0, std::min(rows, half), out);
    append_prefetch_rows(w, half, std::min(rows, half), out);
}

const Weight* single_projection(const ops::ProjectionWeights& projection) noexcept {
    const auto* single = std::get_if<ops::SingleProjectionWeight>(&projection);
    return single != nullptr ? &single->weight : nullptr;
}

std::size_t exchange_prefetch_budget(const ops::PeerEvents& events,
                                     const ExecutionContext& execution,
                                     std::size_t exchange_bytes) {
    if (!prefetch_enabled()) { return 0; }
    const ScopedCurrentDevice restore; // captured_mailbox_for() changes the current device
    const ops::PeerMailbox* mailbox =
        ops::detail::captured_mailbox_for(exchange_bytes, execution, events);
    return mailbox != nullptr ? mailbox->l2_prefetch_budget() : 0;
}

void stage_exchange_prefetch(const ops::PeerEvents& events, const ExecutionContext& execution,
                             std::size_t exchange_bytes,
                             const std::array<ExchangePrefetch, 2>& hint) {
    const ScopedCurrentDevice restore; // captured_mailbox_for() changes the current device
    ops::PeerMailbox* mailbox =
        ops::detail::captured_mailbox_for(exchange_bytes, execution, events);
    if (mailbox == nullptr || mailbox->l2_prefetch_budget() == 0) { return; }
    for (int rank = 0; rank < 2; ++rank) {
        const auto& ranges = hint[static_cast<std::size_t>(rank)];
        const std::size_t count =
            std::min<std::size_t>(ranges.size(), ops::PeerMailbox::kL2PrefetchRanges);
        mailbox->set_l2_prefetch(rank, {ranges.data(), count});
    }
}

} // namespace ninfer::models::qwen3_5::execution
