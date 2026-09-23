#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "core/device.h"
#include "ninfer/ops/allreduce.h"

#include <array>

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters,
                                                       const GdnConfig& config, std::int32_t batch,
                                                       std::int32_t first_width,
                                                       std::int32_t last_width);
[[nodiscard]] std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters,
                                                     const GdnConfig& config, std::int32_t batch,
                                                     std::int32_t first_width,
                                                     std::int32_t last_width);
void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream);
void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      WorkspaceArena& workspace, DeviceExecutionView execution);
void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream);
void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);

// Tensor-parallel forms over two ranks. `parameters[r]` is rank r's shard: key heads
// [Hk/2*r, +Hk/2) and value heads [Hv/2*r, +Hv/2) of every Q|K|V|Z section in one contiguous
// parent, the matching convolution channels, and those value heads' A/B rows, A_log and dt_bias.
// The input RMSNorm is replicated elementwise work and runs per rank before gdn_control_split,
// because the fused norm+gating Op has no split form. The GDN core then runs per rank on the
// rank's heads with no communication. Only the single-parent input projection has a split route.
[[nodiscard]] std::size_t gdn_control_split_workspace_bytes(const GdnConfig& shard,
                                                            std::int32_t hidden, std::int32_t first,
                                                            std::int32_t last);
[[nodiscard]] std::size_t gdn_projection_split_workspace_bytes(const GdnParameters& parameters,
                                                               std::int32_t first,
                                                               std::int32_t last);
[[nodiscard]] std::size_t gdn_snapshot_split_workspace_bytes(const GdnParameters& parameters,
                                                             std::int32_t batch,
                                                             std::int32_t first_width,
                                                             std::int32_t last_width);
void gdn_control_split(const std::array<Tensor, 2>& hidden,
                       const std::array<const GdnParameters*, 2>& parameters,
                       const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta,
                       const std::array<WorkspaceArena*, 2>& workspace,
                       const ExecutionContext& execution);
void gdn_projection_split(const std::array<Tensor, 2>& hidden,
                          const std::array<const GdnParameters*, 2>& parameters,
                          const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                          const std::array<WorkspaceArena*, 2>& workspace,
                          const ExecutionContext& execution);
void gdn_projection_snapshot_split(
    const std::array<Tensor, 2>& hidden, const std::array<const GdnParameters*, 2>& parameters,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_slots, const std::array<Tensor, 2>& destination_slots,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& key,
    const std::array<Tensor, 2>& value, const std::array<Tensor, 2>& z,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution);
// Row-parallel output projection added to the replicated residual; the layer's first all-reduce.
void gdn_output_split(const std::array<Tensor, 2>& normalized,
                      const std::array<const GdnParameters*, 2>& parameters,
                      const std::array<Tensor, 2>& residual, const std::array<Tensor, 2>& staging,
                      const std::array<WorkspaceArena*, 2>& workspace,
                      const ExecutionContext& execution, const ops::PeerEvents& events);

} // namespace ninfer::models::qwen3_5::execution
