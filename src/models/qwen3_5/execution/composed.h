#pragma once

// Composed projection routes for geometries the fused Op catalogs do not register.
//
// The fused Ops (attn_input_proj, gdn_input_proj*, linear_swiglu, linear_add,
// gdn_norm_gating_proj) are closed catalogs of the shapes that have been tuned. A checkpoint
// with a different width (Qwen3.5-9B: hidden 4096, 16/4 attention heads, 32 GDN value heads)
// runs the same mathematics through plain `linear`, `rmsnorm`, `silu_mul`, `residual_add`,
// `gdn_gating` and the family's projected causal-convolution launcher instead. Slower, untuned,
// and correct; `registered_*` says which route a block takes, decided once at parameter
// preparation from the artifact geometry.

#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// Catalog membership of the fused routes for this text geometry.
[[nodiscard]] bool registered_attention_projection(const TextConfig& config);
[[nodiscard]] bool registered_gdn_projection(const TextConfig& config);
[[nodiscard]] bool registered_gdn_control(const TextConfig& config);
[[nodiscard]] bool registered_swiglu(std::int32_t gate_up_rows, std::int32_t k);
[[nodiscard]] bool registered_linear_add(std::int32_t rows, std::int32_t k);
[[nodiscard]] bool registered_pair(std::int32_t rows, std::int32_t k);
// mtp_split_attn_in packs exactly the 27B attention rows (24 x 256 | 4 x 256 | 24 x 256 | 4 x 256).
[[nodiscard]] bool registered_mtp_split(const AttentionConfig& config);

// `ops::linear` takes two-dimensional operands; view any contiguous activation as [rows, cols].
[[nodiscard]] Tensor as_matrix(const Tensor& tensor, std::int32_t rows);

// True when the given logical rows are consecutive regions of one encoded parent: the native
// single-matrix form `ops::prepare_linear_weight(span)` accepts.
[[nodiscard]] bool rows_share_one_parent(std::span<const ops::WeightInput> inputs);

void composed_linear(const Tensor& x, const LinearParameters& p, Tensor& out,
                     WorkspaceArena& workspace, cudaStream_t stream);
void composed_linear_add(const Tensor& x, const LinearParameters& p, Tensor& residual,
                         WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t composed_linear_add_workspace_bytes(const LinearParameters& p,
                                                              std::int32_t first,
                                                              std::int32_t last);

void composed_attention_projection(const Tensor& hidden, const ComposedAttentionProjection& p,
                                   Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                   WorkspaceArena& workspace, cudaStream_t stream);

void composed_gdn_projection(const Tensor& hidden, const ComposedGdnProjection& p, Tensor& qkv,
                             Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t composed_gdn_projection_workspace_bytes(const ComposedGdnProjection& p,
                                                                  std::int32_t first,
                                                                  std::int32_t last);
[[nodiscard]] std::size_t composed_gdn_snapshot_workspace_bytes(const ComposedGdnProjection& p,
                                                                std::int32_t batch,
                                                                std::int32_t first_width,
                                                                std::int32_t last_width);
void composed_gdn_snapshot(const Tensor& hidden, const ComposedGdnProjection& p,
                           const Tensor& convolution, Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           const Tensor& destination_slots, Tensor& query, Tensor& key,
                           Tensor& value, Tensor& z, WorkspaceArena& workspace,
                           cudaStream_t stream);
void composed_gdn_record(const Tensor& hidden, const ComposedGdnProjection& p,
                         const Tensor& convolution, const Tensor& conv_states,
                         const Tensor& valid_columns, const Tensor& initial_slots,
                         Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                         WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t composed_gdn_control_workspace_bytes(const ComposedGdnControl& p,
                                                               std::int32_t first,
                                                               std::int32_t last);
void composed_gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                               const ComposedGdnControl& p, const Tensor& a_log,
                               const Tensor& dt_bias, Tensor& hidden, Tensor& g, Tensor& beta,
                               WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
