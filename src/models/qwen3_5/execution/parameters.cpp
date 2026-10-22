#include "models/qwen3_5/execution/parameters.h"

#include "core/weight_view.h"
#include "runtime/contract/yarn.h"
#include "ninfer/ops/linear.h"

#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

bool contiguous(std::span<const ops::WeightInput> inputs) {
    const WeightParent* parent = nullptr;
    std::uint64_t end          = 0;
    for (const auto& input : inputs) {
        for (const auto& part : input.weight.parts) {
            if (parent && (part.parent != parent || part.begin != end)) { return false; }
            parent = part.parent;
            end    = part.end;
        }
    }
    return parent != nullptr;
}

class Prepare {
public:
    Prepare(const Model& model, int device)
        : model_(model), device_(device), sharded_(model.device_count() > 1) {}

    const BoundWeight& weight(WeightId id) const { return model_.weight(id, device_); }

    bool resident(WeightId id) const { return model_.resident(id, device_); }

    ops::WeightInput input(WeightId id) const { return model_.input(id, device_); }

    ops::WeightInput input(WeightUseId id) const { return model_.input(id, device_); }

    LinearParameters linear(WeightId id) const {
        return with_context(weight(id).name, [&] { return ops::prepare_linear_weight(input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(input(id)); });
    }

    // ops::prepare_{attn,gdn}_input_proj_weights and prepare_gdn_gating_proj_weights name the
    // complete tp1 geometries. A tensor-parallel rank keeps the same row order and native forms
    // over its shard parents: one contiguous parent, or a pair split after `pair` inputs. Format
    // support for the shard geometry is enforced by the split Op that consumes it.
    ops::ProjectionWeights shard_bank(std::span<const ops::WeightInput> inputs,
                                      std::size_t pair) const {
        if (contiguous(inputs)) { return ops::prepare_linear_weight(inputs); }
        return ops::PairedProjectionWeights{
            ops::prepare_linear_weight(inputs.first(pair)).weight,
            ops::prepare_linear_weight(inputs.subspan(pair)).weight};
    }

    ops::ProjectionWeights attention_projection(const AttentionWeights& a) const {
        if (sharded_) {
            const std::array inputs{input(a.query), input(a.key), input(a.gate), input(a.value)};
            return shard_bank(inputs, 2);
        }
        return ops::prepare_attn_input_proj_weights(input(a.query), input(a.key), input(a.gate),
                                                    input(a.value));
    }

    ops::ProjectionWeights gdn_projection(const GdnWeights& g) const {
        if (sharded_) {
            const std::array inputs{input(g.query), input(g.key), input(g.value), input(g.z)};
            return shard_bank(inputs, 2);
        }
        return ops::prepare_gdn_input_proj_weights(input(g.query), input(g.key), input(g.value),
                                                   input(g.z));
    }

    ops::ProjectionWeights gdn_control(const GdnWeights& g) const {
        if (sharded_) {
            const std::array inputs{input(g.a_projection), input(g.b_projection)};
            return shard_bank(inputs, 1);
        }
        return ops::prepare_gdn_gating_proj_weights(input(g.a_projection), input(g.b_projection));
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(
                    weight(w.gate).name,
                    [&] { return ops::prepare_linear_swiglu_weight(input(w.gate), input(w.up)); }),
                linear(w.down)};
    }

    FfnParameters ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
        const auto& moe = std::get<MoeWeights>(w.ffn);
        if (std::get<MoeConfig>(model_.config().text.ffn).num_experts_per_tok != 8) {
            throw std::invalid_argument("SparseMoe implements top-8 routing");
        }
        std::vector<ops::WeightInput> gate_up, down;
        gate_up.reserve(2 * moe.experts.size());
        down.reserve(moe.experts.size());
        for (const auto& expert : moe.experts) {
            gate_up.push_back(input(expert.gate));
            gate_up.push_back(input(expert.up));
            down.push_back(input(expert.down));
        }
        return with_context(weight(moe.router).name, [&] {
            return ops::prepare_sparse_moe_weights(input(moe.router), input(moe.shared_score),
                                                   gate_up, down, input(moe.shared.gate),
                                                   input(moe.shared.up), input(moe.shared.down));
        });
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            out.mixer = AttentionParameters{attention_projection(*a), tensor(a->query_norm),
                                            tensor(a->key_norm), linear(a->output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = GdnParameters{.projection  = gdn_projection(g),
                                          .control     = gdn_control(g),
                                          .a_log       = tensor(g.a_log),
                                          .dt_bias     = tensor(g.dt_bias),
                                          .convolution = tensor(g.convolution),
                                          .norm        = tensor(g.norm),
                                          .output      = linear(g.output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    ops::SparseMoeHints prefetch(const ops::ProjectionWeights& projection, WeightId query) const {
        const auto* single = std::get_if<LinearParameters>(&projection);
        const auto& weight =
            single ? single->weight : std::get<ops::PairedProjectionWeights>(projection).first;
        const auto& geometry = model_.weight(query, device_).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{input(a.query), input(a.key), input(a.gate), input(a.value)};
        MtpParameters out;
        out.input_projection    = linear(w.input_projection);
        out.embedding_norm      = tensor(w.embedding_norm);
        out.hidden_norm         = tensor(w.hidden_norm);
        out.input_norm          = tensor(w.layer.input_norm);
        out.post_attention_norm = tensor(w.layer.post_attention_norm);
        out.final_norm          = tensor(w.final_norm);
        out.projection.packed   = ops::prepare_linear_weight(inputs);
        if (model_.config().text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        }
        out.query_norm = tensor(a.query_norm);
        out.key_norm   = tensor(a.key_norm);
        out.output     = linear(a.output);
        out.ffn        = ffn(w.layer);
        if (resident(w.output_head)) { out.output_head = linear(w.output_head_use); }
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{input(layer.query), input(layer.key), input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context(
                std::string(model_.options().speculative_component()) + "/layers/" +
                    std::to_string(i),
                [&] {
                    const auto& layer = w.layers[i];
                    const auto& a     = layer.attention;
                    DraftBlockParameters result;
                    result.input_norm          = tensor(layer.input_norm);
                    result.post_attention_norm = tensor(layer.post_attention_norm);
                    result.query_key_value     = ops::prepare_attn_input_proj_weights(
                        input(a.query), input(a.key), input(a.value));
                    result.context_key   = linear(a.context_key);
                    result.context_value = linear(a.context_value);
                    result.query_norm    = tensor(a.query_norm);
                    result.key_norm      = tensor(a.key_norm);
                    result.output        = linear(a.output);
                    result.mlp           = dense(layer.mlp);
                    if (layer.attention_conv) {
                        result.attention_conv = convolution(*layer.attention_conv);
                    }
                    if (layer.mlp_conv) { result.mlp_conv = convolution(*layer.mlp_conv); }
                    return result;
                }));
        }
        // The split drafter's selector is held by one rank (load/sharding.h: SingleDevice(1) by
        // default); the other rank's DraftParameters carries none, and the round reads it through
        // the selecting rank's Parameters.
        if (w.selector && resident(w.selector->hidden_projection)) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

private:
    const Model& model_;
    int device_   = 0;
    bool sharded_ = false;
};

} // namespace

Parameters::Parameters(const Model& source, const YarnOptions& yarn, int rank)
    : model(source), device(rank) {
    if (yarn.factor > 1.0F) {
        const auto& text = source.config().text;
        // The loaded artifact is where the rotary shape is known, so the YaRN domain contract
        // (runtime/contract/yarn.h, include/ninfer/ops/rope.h) is enforced here: a factor that the
        // artifact's rotary domain cannot express is reported with its reason at load time instead
        // of being built here and then ignored by the kernel that owns that domain.
        if (!text.rope_parameters || !text.attention) {
            throw std::invalid_argument(
                "--rope-yarn-factor requires a rotary text domain; this artifact has no text "
                "rope_parameters/attention configuration");
        }
        runtime::validate_yarn_rotary_domain(text.attention->head_dim,
                                             text.rope_parameters->rotary_dim, yarn.factor);
        rope_scaling = ops::make_text_yarn_scaling(yarn.factor, yarn.original_context,
                                                  text.rope_parameters->rope_theta);
    }
    // One weight set per tensor-parallel rank: the binding reads rank's device placement.
    const Prepare prepare(model, device);
    const auto& w        = model.weights();
    text.token_embedding = native_weight(prepare.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    text.final_norm      = prepare.tensor(w.text.final_norm);
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i]); }));
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision && prepare.resident(w.vision->patch_embedding)) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft && prepare.resident(w.draft->feature_projection)) {
        draft = with_context(std::string(model.options().speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal && prepare.resident(w.proposal->head)) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids && prepare.resident(*w.proposal->token_ids)) {
            proposal->token_ids = prepare.tensor(*w.proposal->token_ids);
        }
        if (proposal->split()) {
            // A format without the half's route would fail only at the first proposal.
            const auto& head = proposal->head;
            try {
                (void)ops::linear_workspace_capacity_bytes(head.weight.qtype, head.weight.n,
                                                           head.weight.k, head.policy, 1, 1);
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument(
                    "proposal/head: the tensor-parallel half [" + std::to_string(head.weight.n) +
                    "," + std::to_string(head.weight.k) + "] has no registered linear route (" +
                    error.what() + "); NINFER_TP_DRAFT_HEAD=primary keeps it whole on rank 0");
            }
        }
    }
}

} // namespace ninfer::models::qwen3_5::execution
