#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    // Empty on a tensor-parallel rank that does not hold the selected head: a PrimaryOnly
    // optimized proposal head proposes on rank 0 alone. A vocabulary-split one is this rank's
    // row block on both ranks.
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head; // This rank's rows: all of them, or a vocabulary block when split.
    std::optional<Tensor> token_ids; // Row -> token ID map; rank 0 only at tp 2.
    std::uint32_t rows = 0;          // Rows of the whole head.

    // At tp 2 under MTP each rank holds half of the rows (load/sharding.h).
    [[nodiscard]] bool split() const noexcept {
        return static_cast<std::uint64_t>(head.weight.n) != rows;
    }
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
//
// A tensor-parallel Model has one Parameters per rank. Its operands are that rank's shards,
// with head, intermediate and vocabulary extents divided by tp (load/sharding.h); Vision, draft
// and proposal parameters exist only on the rank that holds them.
class Parameters {
public:
    // `rank` selects which tensor-parallel device's weight set this Parameters binds; the
    // per-parameter placement (rows, columns, vocabulary) is in the artifact binding, not here.
    explicit Parameters(const Model& source, const YarnOptions& yarn, int rank = 0);

    // One rank of a two-device load, with native rope (YarnOptions{} == factor 1).
    explicit Parameters(const Model& source, int rank) : Parameters(source, YarnOptions{}, rank) {}
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    const int device;

    // Engine-selected static YaRN coefficients for text/MTP D256/R64 rotation. Disabled (factor 1)
    // preserves native RoPE; Vision and the DFlash draft geometry always stay native.
    ops::TextRopeScaling rope_scaling;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
};

} // namespace ninfer::models::qwen3_5::execution
