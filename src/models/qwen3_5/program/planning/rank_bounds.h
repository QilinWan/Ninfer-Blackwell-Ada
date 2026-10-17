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

// The per-rank general workspace: the phases both ranks run contribute the same maximum, while a
// phase only one rank runs contributes to that rank alone.  Returning both shares is what keeps
// rank 1 from being sized by rank 0's gather (the causal score tile) and from carrying the Vision
// encode workspace it never touches.
struct RankGeneralCapacity {
    std::size_t rank0 = 0;
    std::size_t rank1 = 0;
};

[[nodiscard]] RankGeneralCapacity split_general_capacity(std::size_t shared_capacity,
                                                          std::size_t rank0_only_capacity,
                                                          std::size_t rank1_only_capacity) noexcept;

} // namespace ninfer::models::qwen3_5::detail
