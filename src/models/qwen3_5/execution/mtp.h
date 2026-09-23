#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"

#include <array>

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                                 const AttentionConfig& config, std::int32_t first,
                                                 std::int32_t last);
[[nodiscard]] std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                                         const AttentionConfig& config,
                                                         std::int32_t first, std::int32_t last);
void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream);
void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream);

// Tensor-parallel forms over two ranks. `parameters[r]` is rank r's shard: half of the query/gate
// and of the KV heads, whose packed Q|K|Gate|V rows form one contiguous parent on the rank
// ([7168,5120] for the 27B) and `shard` names that rank's heads. Every form projects through the
// packed parent column-parallel and splits each rank's output with mtp_split_attn_in; the separate
// `rows` projections the single-device incremental path uses are not read, so the KV-only and
// query/gate-only forms compute and discard the other two sections. No communication.
[[nodiscard]] std::size_t
mtp_projection_split_workspace_bytes(const MtpProjectionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
[[nodiscard]] std::size_t mtp_kv_split_workspace_bytes(const MtpProjectionParameters& parameters,
                                                       const AttentionConfig& shard,
                                                       std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t
mtp_query_gate_split_workspace_bytes(const MtpProjectionParameters& parameters,
                                     const AttentionConfig& shard, std::int32_t first,
                                     std::int32_t last);
void mtp_projection_split(const std::array<Tensor, 2>& hidden,
                          const std::array<const MtpProjectionParameters*, 2>& parameters,
                          const AttentionConfig& shard, const std::array<Tensor, 2>& query,
                          const std::array<Tensor, 2>& gate, const std::array<Tensor, 2>& key,
                          const std::array<Tensor, 2>& value,
                          const std::array<WorkspaceArena*, 2>& workspace,
                          const ExecutionContext& execution);
void mtp_kv_projection_split(const std::array<Tensor, 2>& hidden,
                             const std::array<const MtpProjectionParameters*, 2>& parameters,
                             const AttentionConfig& shard, const std::array<Tensor, 2>& key,
                             const std::array<Tensor, 2>& value,
                             const std::array<WorkspaceArena*, 2>& workspace,
                             const ExecutionContext& execution);
void mtp_query_gate_projection_split(
    const std::array<Tensor, 2>& hidden,
    const std::array<const MtpProjectionParameters*, 2>& parameters, const AttentionConfig& shard,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& gate,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution);

// Row-parallel MTP input projection: `input[0]` is the normalized token embedding [H,T] that
// rank 0's column half contracts and `input[1]` the normalized target hidden that rank 1's half
// contracts; one all-reduce leaves the complete [H,T] result in `output[r]`.
void mtp_input_projection_split(const std::array<Tensor, 2>& input,
                                const std::array<const MtpParameters*, 2>& parameters,
                                const std::array<Tensor, 2>& output,
                                const std::array<Tensor, 2>& staging,
                                const std::array<WorkspaceArena*, 2>& workspace,
                                const ExecutionContext& execution, const ops::PeerEvents& events);
// Row-parallel MTP attention output projection over each rank's heads. Unlike the text layers it
// does not add the residual: the caller adds `output[r]`, as the single-device MTP does.
void mtp_output_split(const std::array<Tensor, 2>& attention,
                      const std::array<const MtpParameters*, 2>& parameters,
                      const std::array<Tensor, 2>& output, const std::array<Tensor, 2>& staging,
                      const std::array<WorkspaceArena*, 2>& workspace,
                      const ExecutionContext& execution, const ops::PeerEvents& events);

} // namespace ninfer::models::qwen3_5::execution
