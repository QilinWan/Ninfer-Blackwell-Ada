#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <span>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

// One KV capacity for every rank of a tensor-parallel group. The ranks address the same Main KV
// pages at the same physical indices (rank 0's pools drive rank 1's mirrors), so they must agree
// on one page count, and every rank reserves the same per-rank layout described by `curve`. The
// capacity is resolved against the tightest rank's own free memory, so no rank is overcommitted.
// A single budget reduces to resolve_kv_capacity.
[[nodiscard]] KvCapacityResolution
resolve_kv_capacity_symmetric(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                              std::span<const std::size_t> available_runtime_bytes_per_rank);

} // namespace ninfer::runtime
