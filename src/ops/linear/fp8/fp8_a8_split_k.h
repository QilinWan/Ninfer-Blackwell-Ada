#pragma once
#include "ops/linear/fp8/fp8_schedule.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Capacity bookkeeping for the A8 TMA split-K routes.
//
// A split-K schedule reserves `kPartialBytes` (one resident wave of `kSplitWaveCtas` CTAs times its
// output tile), and the kernel uses it whenever its grid leaves an underfilled final wave: with
// `row_tiles` weight-row tiles and a `kBlockTokens`-token tile, a call at T occupies
// `row_tiles * ceil(T / kBlockTokens)` CTAs, and the schedule splits K when that count modulo its
// wave is non-empty and at most half a wave. Plain (non split-K) TMA and MMA routes never reserve
// partials, so they answer 0.
//
// The wave arithmetic belongs to the schedule, the row-tile count to the shape, and the token band
// to the ladder, so a shape's declared interval capacity is the largest partials requirement of
// every route band it contains. Half of a wave is a *tuning* choice baked into each ladder, not a
// property of a family: the same schedule splits at T=385 on [14336,5120] and at T=193 on its
// output-row half, because the shard halves its own row-tile count.
inline constexpr std::size_t fp8_a8_split_ctas(std::size_t row_tiles, std::size_t block_tokens,
                                               std::size_t wave_ctas, std::int32_t tokens) {
    const std::size_t tiles =
        row_tiles * ((static_cast<std::size_t>(tokens) + block_tokens - 1U) / block_tokens);
    const std::size_t tail = tiles % wave_ctas;
    return tail != 0 && tail <= wave_ctas / 2 ? tiles : 0;
}

// Partials the route needs for one T: `kPartialBytes` when its grid splits K at that T, else 0.
template <class Schedule>
constexpr std::size_t fp8_a8_partial_bytes_at(std::size_t row_tiles, std::int32_t tokens) {
    if constexpr (Schedule::kSplitWaveCtas > 0) {
        if (fp8_a8_split_ctas(row_tiles, static_cast<std::size_t>(Schedule::kBlockTokens),
                              static_cast<std::size_t>(Schedule::kSplitWaveCtas), tokens) != 0) {
            return static_cast<std::size_t>(Schedule::kPartialBytes);
        }
    }
    return 0;
}

// The largest requirement of one route band: partials for a T in [band_begin, band_end] at which
// that split-K schedule actually engages.
template <class Schedule>
constexpr std::size_t fp8_a8_band_partial_bytes(std::size_t band_begin, std::size_t band_end,
                                                std::size_t row_tiles) {
    if constexpr (Schedule::kSplitWaveCtas > 0) {
        for (std::size_t tokens = band_begin; tokens <= band_end; ++tokens) {
            const std::size_t bytes = fp8_a8_partial_bytes_at<Schedule>(
                row_tiles, static_cast<std::int32_t>(tokens));
            if (bytes != 0) return bytes;
        }
    }
    return 0;
}

} // namespace ninfer::ops::detail
