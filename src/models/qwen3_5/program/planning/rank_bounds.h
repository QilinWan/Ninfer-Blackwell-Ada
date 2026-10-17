#pragma once
#include <cstddef>

namespace ninfer::models::qwen3_5::detail {

// The per-rank memory plan, as a pure sum so it is testable without a device or an artifact:
// each rank reserves its own persistent layout plus its own arena share, and one KV page count
// drives both ranks' pools, so the device is budgeted for the larger of the two totals.  Pass
// rank 1's pair as 0, 0 for a single-device plan: the second total is then 0 and never binds.
[[nodiscard]] std::size_t bound_rank_sequence_bytes(std::size_t rank0_persistent_bytes,
                                                    std::size_t rank0_arena_bytes,
                                                    std::size_t rank1_persistent_bytes,
                                                    std::size_t rank1_arena_bytes);

} // namespace ninfer::models::qwen3_5::detail
