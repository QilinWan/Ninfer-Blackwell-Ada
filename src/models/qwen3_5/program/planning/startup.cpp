#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/tp.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/execution/workspace.h"
#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

using execution::dimension;
namespace workspace = execution::workspace;

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

// Per-device CUDA Graph allowances at tp 2, per topology class and captured batch size; each rank
// holds its half of one dual-device graph per class. They are max(3 x observed, 8 MiB), where
// observed is the per-rank free memory prepare_graphs() consumed from before the warm-up (module
// loading included) to after the graph upload, measured on two RTX 5070 Ti at 32K context, max
// concurrency 1 and INT8 KV (2026-09):
//   ordinary:                  2.0 / 2.0 MiB on rank 0 / 1 for its one class -> 8 MiB;
//   MTP3:                      2.0 / 2.0 MiB for its one class -> 8 MiB;
//   DFlash2 K=4 --lm-head-draft: 18.0 / 12.0 MiB for five classes (rank 0 also holds the
//                              drafter), 3.6 MiB per class -> 10.8, rounded up to 11 MiB.
// Every batch size up to max concurrency is warmed and captured, so the allowances scale linearly
// with it; concurrency above 1 is not measured. The server logs observed against allowance per
// rank and warns if a rank exceeds it.
constexpr std::size_t kTp2OrdinaryGraphAllowance     = 8ULL * kMiB;
constexpr std::size_t kTp2MtpGraphClassAllowance     = 8ULL * kMiB;
constexpr std::size_t kTp2DFlash2GraphClassAllowance = 11ULL * kMiB;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

// Rank `rank_index`'s persistent layout. At tp 2 the ranks differ only in the masked drafter's
// state (its context, pending and prefill features and the DFlash local slots of each StateImage):
// the drafter runs on rank 0 alone, so rank 1 holds none of it.
PersistentLayout persistent_layout(const SequencePlanImpl& plan, int rank_index) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;
    // One rank's KV heads and GDN channels/value heads; the whole config at tp 1. The round
    // state keeps the complete vocabulary: rank 0 gathers the complete logits and samples there.
    // Rank 1's round state has the same layout, so its logits frames are allocated but unused.
    const TextConfig rank = execution::shard_text_config(config, plan.tp);
    const bool drafter    = plan.features.masked_draft() && rank_index == 0;

    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots,
        "Qwen3.5 StateImage slot count exceeds int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_5::plan_decoder_state(
        builder, qwen3_5::DecoderStateSpec{
                     .full_attention_layers     = config.full_attention_layers,
                     .mtp_layers                = 1,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = dimension(rank.attention->num_key_value_heads),
                     .attention_head_dim        = dimension(config.attention->head_dim),
                     .kv_storage                = plan.kv_storage,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                 });
    qwen3_5::StateImageSpec state_image_spec{
        .linear =
            {
                .layers         = config.linear_attention_layers,
                .conv_channels  = (rank.gdn ? dimension(rank.gdn->conv_channels()) : 0),
                .conv_width     = (rank.gdn ? dimension(rank.gdn->linear_conv_kernel_dim - 1) : 0),
                .value_heads    = (rank.gdn ? dimension(rank.gdn->linear_num_value_heads) : 0),
                .value_head_dim = (rank.gdn ? dimension(rank.gdn->linear_value_head_dim) : 0),
                .key_head_dim   = (rank.gdn ? dimension(rank.gdn->linear_key_head_dim) : 0),
                .slot_count     = state_image_slots,
                .conv_dtype     = DType::BF16,
            },
        .hidden = dimension(config.hidden_size),
    };
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (drafter) {
            state_image_spec.dflash_local = qwen3_5::DFlashLocalStateSpec{
                .layers   = draft->local_layer_count(),
                .capacity = draft->sliding_window.value_or(0),
                .kv_heads = dimension(draft->attention.num_key_value_heads),
                .head_dim = dimension(draft->attention.head_dim),
            };
        }
    }
    out.state_images = qwen3_5::plan_state_image_device_pool(builder, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder,
            GdnReplayRecordSpec{
                .layers          = dimension(config.linear_attention_layers),
                .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                // One rank's key/value heads and convolution channels, as for the StateImages.
                .conv_channels = (rank.gdn ? dimension(rank.gdn->conv_channels()) : 0),
                .qk_heads      = (rank.gdn ? dimension(rank.gdn->linear_num_key_heads) : 0),
                .value_heads   = (rank.gdn ? dimension(rank.gdn->linear_num_value_heads) : 0),
                .key_dim       = (rank.gdn ? dimension(rank.gdn->linear_key_head_dim) : 0),
                .value_dim     = (rank.gdn ? dimension(rank.gdn->linear_value_head_dim) : 0),
            });
    }
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (drafter) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            if (draft->full_layer_count() != 0) {
                const PagedKVStorageLayout full_storage = paged_kv_storage_layout(
                    KvCacheStorage::BFloat16, dimension(draft->attention.head_dim));
                KVPageGeometry full_geometry{
                    .page_tokens        = kPagedKVPageSize,
                    .device_plane_order = PagedKVPlaneOrder::HeadMajor,
                    .planes =
                        {
                            {full_storage.key.data_dtype, full_storage.key.data_leading_extent,
                             dimension(draft->attention.num_key_value_heads), 256},
                            {full_storage.value.data_dtype, full_storage.value.data_leading_extent,
                             dimension(draft->attention.num_key_value_heads), 256},
                        },
                };
                const auto planes = full_geometry.planes;
                for (std::uint32_t layer = 1; layer < draft->full_layer_count(); ++layer) {
                    full_geometry.planes.insert(full_geometry.planes.end(), planes.begin(),
                                                planes.end());
                }
                dflash.full = qwen3_5::PagedKVCacheLayout{
                    .pages = plan_device_kv_page_pool(
                        builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages,
                                                      .geometry = std::move(full_geometry)}),
                    .execution_tables = plan_kv_execution_tables(
                        builder,
                        KVExecutionTableSpec{
                            .logical_page_capacity = logical_pages,
                            .table_rows = static_cast<std::int32_t>(plan.max_concurrency),
                        }),
                    .layers        = draft->full_layer_count(),
                    .max_context   = plan.capacity,
                    .kv_heads      = dimension(draft->attention.num_key_value_heads),
                    .layer_storage = full_storage,
                };
            }
            dflash.prefill_features = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                 effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                  static_cast<std::int32_t>(plan.draft_window + 1U),
                  static_cast<std::int32_t>(plan.max_concurrency)},
                "DFlash pending target features");
        }
    }

    out.round = qwen3_5::begin_round_state_layout(
        builder, qwen3_5::RoundStateSpec{.hidden         = dimension(config.hidden_size),
                                         .output_rows    = dimension(config.vocab_size),
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .backend        = plan.speculative_backend,
                                         .causal_scoring = plan.causal_scoring});
    out.prefill_hidden =
        add_tensor(builder, DType::BF16, {dimension(config.hidden_size), effective_prefill_chunk},
                   "step prefill hidden");
    if (plan.causal_scoring) {
        out.score_hidden =
            add_tensor(builder, DType::BF16,
                       {dimension(config.hidden_size), static_cast<std::int32_t>(kCausalScoreTile)},
                       "causal score hidden staging");
    }
    qwen3_5::complete_round_state_layout(builder, out.round);
    if (!plan.causal_scoring) {
        out.grammar_masks =
            add_tensor(builder, DType::I32,
                       {dimension((parameters.model.resources().public_token_count + 31) / 32),
                        static_cast<std::int32_t>(plan.draft_window + 1),
                        static_cast<std::int32_t>(plan.max_concurrency)},
                       "grammar token masks");
        out.token_counts        = add_tensor(builder, DType::I32,
                                             {dimension(parameters.model.resources().public_token_count),
                                              static_cast<std::int32_t>(plan.max_concurrency)},
                                             "sampling token counts");
        const auto config_words = static_cast<std::int32_t>(
            (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
        out.sampling_config = add_tensor(
            builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
            "sampling config");
    }
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

void reserve_matrix(WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                    std::int32_t tokens) {
    (void)layout.alloc(dtype, {rows, tokens});
}

void reserve_scratch(WorkspaceLayoutBuilder& layout, std::size_t bytes) {
    if (bytes == 0) { return; }
    auto scope = layout.scope();
    (void)layout.alloc_bytes(bytes);
}

void reserve_linear(WorkspaceLayoutBuilder& layout, const execution::LinearParameters& p, int first,
                    int last) {
    reserve_scratch(layout, ops::linear_workspace_capacity_bytes(
                                p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last));
}

void reserve_linear_add(WorkspaceLayoutBuilder& layout, const execution::LinearParameters& p,
                        int first, int last) {
    reserve_scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last));
}

// The drafter's transient phases (execution/draft.cpp). They run on the executing device at tp 1
// and on rank 0 alone at tp 2, where the drafter and its proposal head are whole; both widths plan
// them from rank 0's Parameters.

// One context append over `batch` rows of `width` target columns; `compact_input` adds the compact
// feature matrix the decode round gathers first.
std::size_t dflash_context_workspace(const SequencePlanImpl& plan, std::int32_t width,
                                     std::int32_t batch, bool compact_input) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;
    const auto& draft      = *parameters.model.config().draft;
    const auto tokens      = width * batch;
    WorkspaceLayoutBuilder layout;
    if (compact_input) {
        reserve_matrix(layout, DType::BF16,
                       dimension(config.hidden_size * std::uint64_t(draft.target_layer_ids.size())),
                       tokens);
    }
    if (draft.dflash2.has_value()) {
        const auto local_width = std::min(width, dimension(draft.sliding_window.value_or(0)));
        (void)workspace::dflash_context(layout, config, draft, local_width * batch);
        reserve_linear(layout, parameters.draft->feature_projection, local_width * batch,
                       local_width * batch);
        reserve_scratch(layout, ops::context_kv_materialize_workspace_capacity_bytes(
                                    batch, local_width, local_width));
        return layout.peak_bytes(1);
    }
    (void)workspace::dflash_context(layout, config, draft, tokens);
    reserve_linear(layout, parameters.draft->feature_projection, tokens, tokens);
    {
        auto layer = layout.scope();
        (void)workspace::dflash_context_layer(layout, config, draft, tokens);
    }
    return layout.peak_bytes(1);
}

// One proposal over `batch` rows of `width` masked-block columns.
std::size_t dflash_proposal_workspace(const SequencePlanImpl& plan, std::int32_t width,
                                      std::int32_t batch) {
    const auto& parameters    = *plan.parameters;
    const auto& config        = parameters.model.config().text;
    const auto& draft         = *parameters.model.config().draft;
    const auto drafts         = static_cast<std::int32_t>(plan.draft_window);
    const std::int32_t tokens = width * batch;
    const auto& head          = plan.proposal_head == ProposalHead::Optimized
                                    ? parameters.proposal->head
                                    : parameters.draft->output_head;
    WorkspaceLayoutBuilder layout;
    reserve_matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
    if (draft.dflash2.has_value()) {
        const auto prepare = [&] {
            (void)workspace::dflash2_branch(layout, config, draft, width, batch);
            reserve_scratch(layout,
                            ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                width, width, batch, batch));
        };
        {
            auto attention = layout.scope();
            prepare();
            reserve_matrix(layout, DType::BF16, dimension(draft.attention.query_width()), tokens);
            reserve_matrix(layout, DType::BF16, dimension(draft.attention.key_width()), tokens);
            reserve_matrix(layout, DType::BF16, dimension(draft.attention.key_width()), tokens);
            reserve_matrix(layout, DType::BF16, dimension(draft.attention.query_width()), tokens);
            reserve_scratch(layout, ops::sliding_window_attention_workspace_capacity_bytes(
                                        {dimension(draft.attention.head_dim),
                                         dimension(draft.attention.num_attention_heads),
                                         dimension(draft.attention.num_key_value_heads)},
                                        dimension(draft.sliding_window.value_or(0)),
                                        {0, plan.capacity}, width, width, batch));
            reserve_scratch(
                layout, ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                            dimension(draft.attention.query_width()), width, width, batch, batch));
        }
        {
            auto mlp = layout.scope();
            prepare();
            reserve_matrix(layout, DType::BF16, dimension(draft.intermediate_size), tokens);
            for (const auto& block : parameters.draft->layers) {
                const auto& p = block.mlp.gate_up;
                reserve_scratch(
                    layout, ops::linear_swiglu_workspace_capacity_bytes(
                                p.weight.qtype, p.weight.n, p.weight.k, p.policy, tokens, tokens));
            }
            reserve_scratch(layout,
                            ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                dimension(draft.intermediate_size), width, width, batch, batch));
        }
        const auto mask_columns = drafts * batch;
        reserve_matrix(layout, DType::BF16, dimension(config.hidden_size), mask_columns);
        reserve_matrix(layout, DType::FP32, dimension(draft.dflash2->selector_top_k), mask_columns);
        reserve_scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                                    head.weight.qtype, head.weight.n, head.weight.k, mask_columns,
                                    mask_columns));
        reserve_matrix(layout, DType::BF16, dimension(draft.dflash2->selector_rank), mask_columns);
        reserve_linear(layout, parameters.draft->selector->hidden_projection, mask_columns,
                       mask_columns);
        reserve_scratch(layout, ops::candidate_selector_path_workspace_capacity_bytes(
                                    drafts, drafts, batch, batch));
        return layout.peak_bytes(1);
    }
    {
        auto attention = layout.scope();
        (void)workspace::dflash_attention(layout, config, draft, tokens);
        reserve_scratch(layout, std::max(ops::sliding_window_attention_workspace_capacity_bytes(
                                             {dimension(draft.attention.head_dim),
                                              dimension(draft.attention.num_attention_heads),
                                              dimension(draft.attention.num_key_value_heads)},
                                             dimension(draft.sliding_window.value_or(0)),
                                             {0, plan.capacity}, width, width, batch),
                                         ops::context_softmax_attention_workspace_capacity_bytes(
                                             {dimension(draft.attention.head_dim),
                                              dimension(draft.attention.num_attention_heads),
                                              dimension(draft.attention.num_key_value_heads)},
                                             {0, plan.capacity}, width, width, batch)));
        for (const auto& block : parameters.draft->layers) {
            reserve_linear_add(layout, block.output, tokens, tokens);
        }
    }
    {
        auto mlp = layout.scope();
        (void)workspace::dflash_mlp(layout, config, draft, tokens);
        for (const auto& block : parameters.draft->layers) {
            const auto& p = block.mlp.gate_up;
            reserve_scratch(layout,
                            ops::linear_swiglu_workspace_capacity_bytes(
                                p.weight.qtype, p.weight.n, p.weight.k, p.policy, tokens, tokens));
        }
        for (const auto& block : parameters.draft->layers) {
            reserve_linear_add(layout, block.mlp.down, tokens, tokens);
        }
    }
    reserve_matrix(layout, DType::BF16, dimension(config.hidden_size), drafts * batch);
    reserve_matrix(layout, DType::BF16, dimension(config.hidden_size), drafts * batch);
    if (plan.proposal_head == ProposalHead::Optimized) {
        reserve_matrix(layout, DType::BF16, dimension(parameters.proposal->rows), drafts * batch);
    } else {
        reserve_matrix(layout, DType::BF16, dimension(config.vocab_size), drafts * batch);
    }
    reserve_linear(layout, head, drafts * batch, drafts * batch);
    return layout.peak_bytes(1);
}

// Rank 0's acceptance over `batch` verified rows: sparse for DFlash2's candidate proposals,
// greedy for DFlash.
std::size_t dflash_accept_workspace(const SequencePlanImpl& plan, std::int32_t batch) {
    const auto& parameters   = *plan.parameters;
    const auto drafts        = static_cast<std::int32_t>(plan.draft_window);
    const auto public_tokens = dimension(parameters.model.resources().public_token_count);
    return parameters.model.config().draft->dflash2.has_value()
               ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                     public_tokens, {false}, drafts, drafts, batch, batch)
               : ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                     public_tokens, drafts, drafts, batch, batch);
}

// One rank's transient workspace at tensor-parallel width 2, following the allocation order of
// TextContext's split schedule (execution/text.cpp: prefill_impl_tp2 with its MTP prompt
// alignment, ordinary_decode_batch_tp2, target verification, the MTP forwards and proposals,
// run_layers_tp2 and logits_tp2) and of rank 0's DFlash2 drafter (execution/draft.cpp). Per-layer
// stages use the rank's share of the config and the rank's shard Parameters; the call roots (ids,
// positions, residual, all-reduce staging) keep the replicated hidden width. Both ranks allocate
// this capacity: where the ranks differ (rank 0's logits gather staging, sampling, token
// embedding, optimized proposal head and drafter, rank 1's last hidden column), the plan covers
// both. It is built from rank 0's Parameters, whose shard shapes equal rank 1's except for the
// heads and the drafter only rank 0 holds.
WorkspacePlan build_tensor_parallel_workspace_plan(const SequencePlanImpl& plan) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;
    const TextConfig shard = execution::shard_text_config(config, plan.tp);
    if (plan.causal_scoring || plan.features.vision ||
        plan.speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument(
            "tensor-parallel workspace supports ordinary, MTP and DFlash2 text generation only");
    }
    const bool mtp = plan.speculative_backend == SpeculativeBackend::Mtp;
    if (mtp && (!parameters.mtp || !shard.attention)) {
        throw std::invalid_argument("tensor-parallel MTP workspace requires the MTP parameters");
    }

    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto hidden = dimension(config.hidden_size);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::CausalAttentionExecutionEnvelope text_envelope{1, plan.capacity};
    const std::int32_t public_tokens = dimension(parameters.model.resources().public_token_count);

    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };
    const auto row_parallel_scratch = [&](WorkspaceLayoutBuilder& layout,
                                          const execution::LinearParameters& p, int first,
                                          int last) {
        reserve_scratch(layout, ops::linear_add_row_parallel_workspace_capacity_bytes(
                                    p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last));
    };
    const auto attention_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t batch_size,
                                       std::int32_t min_width, std::int32_t max_width) {
        reserve_scratch(layout,
                        ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(shard.attention->head_dim),
                             dimension(shard.attention->num_attention_heads),
                             dimension(shard.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, batch_size, min_width, max_width,
                            DeviceExecutionView{nullptr, plan.multiprocessor_count}));
    };
    // `record` selects the ReplaySSM record projection of speculative verification instead of the
    // in-place snapshot of ordinary decode.
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, TextPhase phase, bool record,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width) {
        for (const auto& block : parameters.text.layers) {
            {
                auto stage = layout.scope();
                if (const auto* attention =
                        std::get_if<execution::AttentionParameters>(&block.mixer)) {
                    (void)workspace::text_attention_projection(layout, shard, last);
                    reserve_scratch(layout, execution::attention_projection_split_workspace_bytes(
                                                *attention, first, last));
                    (void)workspace::text_attention_results(layout, shard, last);
                    attention_scratch(layout, batch_size, min_width, max_width);
                    row_parallel_scratch(layout, attention->output, first, last);
                } else {
                    const auto& gdn = std::get<execution::GdnParameters>(block.mixer);
                    (void)workspace::gdn_control(layout, shard, last);
                    reserve_scratch(layout, execution::gdn_control_split_workspace_bytes(
                                                *shard.gdn, hidden, first, last));
                    (void)workspace::gdn_projection(layout, shard, last);
                    if (phase == TextPhase::Verify) {
                        reserve_scratch(layout, record
                                                    ? execution::gdn_record_split_workspace_bytes(
                                                          gdn, batch_size, min_width, max_width)
                                                    : execution::gdn_snapshot_split_workspace_bytes(
                                                          gdn, batch_size, min_width, max_width));
                    } else {
                        (void)workspace::gdn_prefill_conv(layout, shard, last);
                        reserve_scratch(layout, execution::gdn_projection_split_workspace_bytes(
                                                    gdn, first, last));
                    }
                    (void)workspace::gdn_recurrent_output(layout, shard, last);
                    if (phase == TextPhase::Prefill) {
                        reserve_scratch(layout, ops::gated_delta_net_workspace_capacity_bytes(
                                                    dimension(shard.gdn->linear_num_key_heads),
                                                    dimension(shard.gdn->linear_num_value_heads),
                                                    first, last));
                    }
                    (void)workspace::gdn_normalized_output(layout, shard, last);
                    row_parallel_scratch(layout, gdn.output, first, last);
                }
            }
            auto stage = layout.scope();
            (void)workspace::post_mixer_hidden(layout, config, last);
            reserve_scratch(layout, execution::ffn_split_workspace_bytes(block.ffn, first, last));
        }
    };
    // Vocabulary-split head over `columns` final hidden columns: this rank's rows, rank 0's
    // staging of rank 1's rows, and the column-parallel projection's scratch.
    const auto split_logits = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        const execution::LinearParameters& head = parameters.text.output_head;
        const std::int32_t peer_rows            = dimension(config.vocab_size) - head.weight.n;
        auto call                               = layout.scope();
        (void)workspace::tp_logits(layout, head.weight.n, peer_rows, columns, true);
        reserve_scratch(layout,
                        execution::output_head_split_workspace_bytes(head, columns, columns));
    };
    // The MTP proposal over `columns` hidden columns: rank 0's optimized head alone, or the
    // vocabulary-split output head.
    const auto proposal = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        auto call = layout.scope();
        if (plan.proposal_head == ProposalHead::Optimized) {
            reserve_matrix(layout, DType::BF16, dimension(parameters.proposal->rows), columns);
            reserve_linear(layout, parameters.proposal->head, columns, columns);
        } else {
            split_logits(layout, columns);
        }
    };
    // TextContext::mtp_forward_core_tp2 over `tokens` columns, as `batch_size` rows of `width`.
    const auto mtp_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              std::int32_t batch_size, std::int32_t width) {
        const execution::MtpParameters& p = *parameters.mtp;
        auto core                         = layout.scope();
        (void)workspace::tp_call_roots(layout, config, tokens);
        (void)workspace::mtp_stem_split(layout, config, tokens, true);
        reserve_linear(layout, p.input_projection, tokens, tokens);
        (void)workspace::mtp_attention_projection(layout, shard, tokens);
        reserve_scratch(
            layout, execution::mtp_projection_split_workspace_bytes(p.projection, tokens, tokens));
        (void)workspace::mtp_attention_results(layout, shard, tokens);
        attention_scratch(layout, batch_size, width, width);
        (void)workspace::mtp_post_attention(layout, config, tokens);
        reserve_linear(layout, p.output, tokens, tokens);
        reserve_scratch(layout, execution::mtp_ffn_split_workspace_bytes(p.ffn, tokens, tokens));
    };
    // TextContext::mtp_prefill_chunk_tp2 over a `tokens`-column chunk, the prompt's last one when
    // `last_chunk`.
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                       bool last_chunk) {
        const execution::MtpParameters& p = *parameters.mtp;
        const auto key_width              = dimension(shard.attention->key_width());
        const auto query_width            = dimension(shard.attention->query_width());
        auto call                         = layout.scope();
        (void)workspace::tp_call_roots(layout, config, tokens);
        if (last_chunk) {
            for (int i = 0; i < 3; ++i) { reserve_matrix(layout, DType::BF16, hidden, 1); }
        }
        {
            auto bulk = layout.scope();
            (void)workspace::mtp_stem_split(layout, config, tokens, true);
            reserve_linear(layout, p.input_projection, tokens, tokens);
            reserve_matrix(layout, DType::BF16, key_width, tokens);
            reserve_matrix(layout, DType::BF16, key_width, tokens);
            reserve_scratch(layout, execution::mtp_kv_split_workspace_bytes(
                                        p.projection, *shard.attention, tokens, tokens));
            reserve_matrix(layout, DType::BF16, key_width, tokens);
        }
        if (!last_chunk) { return; }
        reserve_matrix(layout, DType::BF16, query_width, 1);
        reserve_matrix(layout, DType::BF16, query_width, 1);
        reserve_scratch(layout, execution::mtp_query_gate_split_workspace_bytes(
                                    p.projection, *shard.attention, 1, 1));
        reserve_matrix(layout, DType::BF16, query_width, 1);
        reserve_matrix(layout, DType::BF16, query_width, 1);
        reserve_matrix(layout, DType::BF16, hidden, 1);
        reserve_matrix(layout, DType::BF16, hidden, 1);
        attention_scratch(layout, 1, 1, 1);
        reserve_linear(layout, p.output, 1, 1);
        reserve_scratch(layout, execution::mtp_ffn_split_workspace_bytes(p.ffn, 1, 1));
        proposal(layout, 1);
    };

    WorkspacePlan out;
    // The text prefill chunk up to its sampled token; the MTP prompt alignment continues from it.
    const auto text_prefill = [&](WorkspaceLayoutBuilder& layout) {
        // A text chunk continuing a multimodal prefix carries a one-axis RoPE offset; rank 1
        // holds its own copy of the delta.
        (void)workspace::text_prefill_roots(layout, config, chunk, 1, 0);
        (void)workspace::tp_call_roots(layout, config, chunk);
        (void)layout.alloc(DType::I32, {1});
        target_body(layout, 1, chunk, TextPhase::Prefill, false, 1, 1, chunk);
        reserve_matrix(layout, DType::BF16, hidden, 1);
        split_logits(layout, 1);
        reserve_scratch(layout, ops::sampling_workspace_capacity_bytes(public_tokens, 1, 1));
    };
    {
        WorkspaceLayoutBuilder layout;
        text_prefill(layout);
        out.text_prefill = finish(layout);
    }
    // Speculative verification of `batch` rows of K+1 columns on both ranks, with ReplaySSM
    // records and the gathered logits in rank 0's frame.
    const auto verification = [&](std::int32_t batch) {
        const std::int32_t aggregate = batch * verify;
        WorkspaceLayoutBuilder target;
        reserve_matrix(target, DType::BF16, hidden, aggregate);
        (void)workspace::tp_call_roots(target, config, aggregate);
        target_body(target, aggregate, aggregate, TextPhase::Verify, true, batch, verify, verify);
        {
            auto logits = target.scope();
            const std::int32_t rows = parameters.text.output_head.weight.n;
            (void)workspace::tp_logits(target, rows, dimension(config.vocab_size) - rows, aggregate,
                                       true);
            reserve_scratch(target, execution::output_head_split_workspace_bytes(
                                        parameters.text.output_head, aggregate, aggregate));
        }
        return finish(target);
    };
    if (plan.speculative_backend == SpeculativeBackend::None) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder layout;
            reserve_matrix(layout, DType::BF16, hidden, batch);
            (void)workspace::tp_call_roots(layout, config, batch);
            target_body(layout, batch, batch, TextPhase::Verify, false, batch, 1, 1);
            reserve_matrix(layout, DType::BF16, hidden, batch);
            split_logits(layout, batch);
            reserve_scratch(layout,
                            ops::sampling_workspace_capacity_bytes(public_tokens, batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(layout));
        }
    } else if (mtp) {
        {
            WorkspaceLayoutBuilder layout;
            text_prefill(layout);
            reserve_matrix(layout, DType::I32, 1, chunk);
            {
                auto final_chunk = layout.scope();
                mtp_prefill_chunk(layout, chunk, true);
            }
            // The prompt proposal steps after the final chunk: the next hidden, the rope
            // positions and rank 1's delta, one MTP column and its proposal.
            for (std::int32_t step = 1; step < drafts; ++step) {
                auto ar_step = layout.scope();
                reserve_matrix(layout, DType::BF16, hidden, 1);
                reserve_matrix(layout, DType::I32, 1, 1);
                reserve_matrix(layout, DType::I32, 1, 1);
                mtp_core(layout, 1, 1, 1);
                proposal(layout, 1);
            }
            out.mtp_prefill = finish(layout);
        }
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder alignment;
            mtp_core(alignment, aggregate, batch, verify);
            WorkspaceLayoutBuilder ar;
            mtp_core(ar, batch, batch, 1);
            WorkspaceLayoutBuilder proposals;
            proposal(proposals, batch);
            const std::size_t accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    public_tokens, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, verification(batch), finish(alignment),
                                      finish(ar), finish(proposals), accept});
        }
    } else {
        // DFlash2: rank 0 appends the prefill features to the drafter's context after each chunk's
        // text schedule, and a round runs rank 0's context append and proposal, the two-device
        // verification and rank 0's sparse acceptance, each from a reset arena.
        out.dflash_context = dflash_context_workspace(plan, chunk, 1, false);
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            out.dflash_round = std::max({out.dflash_round, verification(batch),
                                         dflash_accept_workspace(plan, batch),
                                         dflash_context_workspace(plan, verify, batch, true),
                                         dflash_proposal_workspace(plan, verify, batch)});
        }
    }
    out.general_capacity = std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill,
                                     out.mtp_round, out.dflash_context, out.dflash_round});
    out.capacity = out.general_capacity;
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    if (plan.tp != 1) { return build_tensor_parallel_workspace_plan(plan); }
    const DeviceExecutionView device_execution{nullptr, plan.multiprocessor_count};
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;

    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::CausalAttentionExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace::text_prefill_roots(layout, config, tokens, plan.features.vision ? 3 : 0,
                                            plan.features.vision ? tokens : 0);
    };
    const auto linear_scratch = [&](WorkspaceLayoutBuilder& layout,
                                    const execution::LinearParameters& p, int first, int last) {
        scratch(layout, ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                             p.policy, first, last));
    };
    const auto add_scratch = [&](WorkspaceLayoutBuilder& layout,
                                 const execution::LinearParameters& p, int first, int last) {
        scratch(layout, ops::linear_add_workspace_capacity_bytes(
                            p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width,
                                 ops::CausalAttentionExecutionEnvelope envelope) {
        for (const auto& block : parameters.text.layers) {
            {
                auto stage = layout.scope();
                if (const auto* attention =
                        std::get_if<execution::AttentionParameters>(&block.mixer)) {
                    (void)workspace::text_attention_projection(layout, config, last);
                    scratch(layout, execution::attention_projection_workspace_bytes(*attention,
                                                                                    first, last));
                    (void)workspace::text_attention_results(layout, config, last);
                    scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                        {dimension(config.attention->head_dim),
                                         dimension(config.attention->num_attention_heads),
                                         dimension(config.attention->num_key_value_heads)},
                                        plan.kv_storage, envelope, batch_size, min_width, max_width,
                                        device_execution));
                    add_scratch(layout, attention->output, first, last);
                } else {
                    const auto& gdn = std::get<execution::GdnParameters>(block.mixer);
                    (void)workspace::gdn_control(layout, config, last);
                    scratch(layout, ops::gdn_norm_gating_proj_workspace_capacity_bytes(
                                        dimension(config.gdn->linear_num_value_heads),
                                        dimension(config.hidden_size), first, last));
                    (void)workspace::gdn_projection(layout, config, last);
                    if (path == GdnWorkspacePath::Snapshot) {
                        scratch(layout, execution::gdn_snapshot_workspace_bytes(
                                            gdn, *config.gdn, batch_size, min_width, max_width));
                    } else if (path == GdnWorkspacePath::ReplayRecord) {
                        scratch(layout, execution::gdn_record_workspace_bytes(
                                            gdn, *config.gdn, batch_size, min_width, max_width));
                    } else {
                        (void)workspace::gdn_prefill_conv(layout, config, last);
                        scratch(layout,
                                execution::gdn_projection_workspace_bytes(gdn, first, last));
                    }
                    (void)workspace::gdn_recurrent_output(layout, config, last);
                    if (path == GdnWorkspacePath::Prefill) {
                        scratch(layout,
                                ops::gated_delta_net_workspace_capacity_bytes(
                                    dimension(config.gdn->linear_num_key_heads),
                                    dimension(config.gdn->linear_num_value_heads), first, last));
                    }
                    (void)workspace::gdn_normalized_output(layout, config, last);
                    add_scratch(layout, gdn.output, first, last);
                }
            }
            auto stage = layout.scope();
            (void)workspace::post_mixer_hidden(layout, config, last);
            scratch(layout, execution::ffn_workspace_bytes(block.ffn, first, last));
        }
        if (!plan.causal_scoring) {
            linear_scratch(layout, parameters.text.output_head, first, last);
        }
    };
    const auto mtp_post_mixer = [&](WorkspaceLayoutBuilder& layout, int first, int last) {
        linear_scratch(layout, parameters.mtp->output, first, last);
        scratch(layout, execution::ffn_workspace_bytes(parameters.mtp->ffn, first, last, true));
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, dimension(parameters.proposal->rows), columns);
            linear_scratch(layout, parameters.proposal->head, columns, columns);
        } else {
            linear_scratch(layout, parameters.mtp->output_head, columns, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace::mtp_stem(layout, config, tokens, !preembedded);
        linear_scratch(layout, parameters.mtp->input_projection, 1, tokens);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace::mtp_attention_projection(layout, config, tokens);
        scratch(layout, execution::mtp_projection_workspace_bytes(parameters.mtp->projection,
                                                                  tokens, tokens));
        (void)workspace::mtp_attention_results(layout, config, tokens);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, envelope, 1, tokens, tokens, device_execution));
        (void)workspace::mtp_post_attention(layout, config, tokens);
        mtp_post_mixer(layout, tokens, tokens);
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            scratch(layout, execution::mtp_kv_workspace_bytes(parameters.mtp->projection,
                                                              *config.attention, first, last));
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
        }
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, execution::mtp_query_gate_workspace_bytes(parameters.mtp->projection,
                                                                  *config.attention, 1, 1));
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, 1, 1, 1, device_execution));
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        mtp_post_mixer(layout, 1, 1);
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, text_envelope);
    if (!plan.causal_scoring) {
        scratch(text_prefill,
                ops::sampling_workspace_capacity_bytes(
                    dimension(parameters.model.resources().public_token_count), 1, 1));
    }
    out.text_prefill = finish(text_prefill);

    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder causal_score;
        matrix(causal_score, DType::BF16, dimension(config.vocab_size),
               static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::I32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::FP32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        linear_scratch(causal_score, parameters.text.output_head, 1, kCausalScoreTile);
        out.causal_score = finish(causal_score);
    }

    if (!plan.causal_scoring) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder ordinary;
            matrix(ordinary, DType::BF16, dimension(config.hidden_size), batch);
            target_body(ordinary, batch, batch, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::Snapshot, batch, 1, 1, text_envelope);
            scratch(ordinary,
                    ops::sampling_workspace_capacity_bytes(
                        dimension(parameters.model.resources().public_token_count), batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
        }
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), chunk);
            (void)workspace::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            dimension(parameters.model.resources().public_token_count), drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
            target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace::mtp_attention_projection(layout, config, tokens);
                scratch(layout, execution::mtp_projection_workspace_bytes(
                                    parameters.mtp->projection, tokens, tokens));
                (void)workspace::mtp_attention_results(layout, config, tokens);
                scratch(layout,
                        ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, batch, width, width, device_execution));
                (void)workspace::mtp_post_attention(layout, config, tokens);
                mtp_post_mixer(layout, tokens, tokens);
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    dimension(parameters.model.resources().public_token_count), drafts, drafts,
                    batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.masked_draft()) {
        out.dflash_context = dflash_context_workspace(plan, chunk, 1, false);
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = verify * batch;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
            target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
            out.dflash_round =
                std::max({out.dflash_round, finish(target), dflash_accept_workspace(plan, batch),
                          dflash_context_workspace(plan, verify, batch, true),
                          dflash_proposal_workspace(plan, verify, batch)});
        }
    }

    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.causal_score});
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        const std::uint32_t merged = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(plan.capacity, kMaximumVisionItemTokens));
        out.vision = execution::VisionContext::plan_workspace(
            *parameters.model.config().vision, *parameters.vision, merged, out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

void validate_target_options(const execution::Parameters& parameters, DeviceContext& device,
                             const EngineOptions& options) {
    if (!parameters.model.config().text.attention ||
        parameters.model.config().text.full_attention_layers == 0) {
        throw std::invalid_argument("Qwen3.5 Program requires at least one full-attention layer");
    }
    if (parameters.model.options() != models::load_options(options)) {
        throw std::invalid_argument(
            "loaded components do not match the requested execution options");
    }
    // The draft shares the YaRN-extended positional frame: it only attends a short sliding
    // window, so the absolute position matters far less than for the target, but its declared
    // native window would otherwise veto a context the target can already serve.  Scale it by the
    // same factor and let the measured acceptance rate decide whether it is still useful.
    const double draft_scale = options.yarn.factor > 1.0F
                                   ? static_cast<double>(options.yarn.factor)
                                   : 1.0;
    if (parameters.draft &&
        static_cast<double>(options.max_context) >
            static_cast<double>(parameters.model.config().draft->max_position_embeddings) *
                draft_scale) {
        throw std::invalid_argument("max_context exceeds the selected draft position capacity");
    }
    // YaRN scales the positional transform, so the usable window is the artifact native window
    // times the factor (runtime::validate_yarn enforces the same bound separately).  Without this
    // the artifact native value would reject a --max-context that YaRN can actually serve.
    const double yarn_scale = options.yarn.factor > 1.0F
                                  ? static_cast<double>(options.yarn.factor)
                                  : 1.0;
    const auto position_capacity = static_cast<std::uint64_t>(
        static_cast<double>(parameters.model.config().text.max_position_embeddings) * yarn_scale);
    if (options.max_context == 0 || options.max_context > position_capacity) {
        throw std::invalid_argument("max_context exceeds the configured position capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash:
    case SpeculativeBackend::DFlash2:
        if (!parameters.draft || (options.speculative.backend == SpeculativeBackend::DFlash2) !=
                                     parameters.model.config().draft->dflash2.has_value()) {
            throw std::invalid_argument(
                "selected masked draft backend is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 15) {
            throw std::invalid_argument("masked draft window must be in [1,15]");
        }
        break;
    }
    // The image is compiled for exactly one architecture, and a cubin only loads on the
    // device family it was built for, so the requirement is the compile-time target.
#ifdef NINFER_SM89
    constexpr int kRequiredComputeCapability = 89;
    constexpr const char* kRequiredDeviceClass =
        "compute capability 8.9 (Ada: RTX 4090 / 4090D / 4080 SUPER / 4070 family)";
#else
    constexpr int kRequiredComputeCapability = 120;
    constexpr const char* kRequiredDeviceClass = "compute capability 12.0 (Blackwell sm_120a)";
#endif
    if (device.compute_capability() != kRequiredComputeCapability) {
        throw std::invalid_argument(std::string("This NInfer build requires ") +
                                    kRequiredDeviceClass);
    }
    if (options.tp != 1 && options.tp != execution::kTensorParallelWidth) {
        throw std::invalid_argument("Qwen3.5 tensor-parallel width must be 1 or 2");
    }
    if (options.tp != 1) {
        if (parameters.model.device_count() != options.tp || parameters.device != 0) {
            throw std::invalid_argument(
                "tensor-parallel planning requires rank 0 Parameters of a two-device Model");
        }
        // The split schedule implements prefill, the ordinary decode round, the MTP round and
        // the DFlash2 round of the dense Text model (TextContext); everything else runs on one
        // device only.
        if (options.purpose != EnginePurpose::Generation ||
            options.speculative.backend == SpeculativeBackend::DFlash || options.enable_vision) {
            throw std::invalid_argument(
                "tensor-parallel execution supports ordinary, MTP and DFlash2 text generation only "
                "(no DFlash, Vision or causal scoring)");
        }
        // The DFlash2 drafter runs whole on rank 0, while the full output head is split by
        // vocabulary rows across the ranks.
        if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
            options.speculative.proposal_head != ProposalHead::Optimized) {
            throw std::invalid_argument(
                "tensor-parallel DFlash2 requires the optimized proposal head (--lm-head-draft)");
        }
        if (options.kv_cache != KvCacheStorage::BFloat16 &&
            options.kv_cache != KvCacheStorage::Int8Group64) {
            throw std::invalid_argument(
                "tensor-parallel attention supports only bf16 and int8 KV caches");
        }
        // This tree merges the Host StateImage/KV/in-flight budget into one knob.
        if (options.context_cache.host_capacity_bytes.value_or(0) != 0) {
            throw std::invalid_argument("tensor-parallel execution requires a Host context budget "
                                        "of 0");
        }
        // Rejects the MoE FFN and extents the width does not divide.
        (void)execution::shard_text_config(parameters.model.config().text, options.tp);
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                  = std::make_unique<SequencePlanImpl>();
    impl->parameters           = inputs.parameters;
    impl->capacity             = inputs.capacity;
    impl->main_page_groups     = main_page_groups;
    impl->kv_capacity          = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency      = inputs.max_concurrency;
    impl->prefill_chunk        = inputs.prefill_chunk;
    impl->draft_window         = inputs.draft_window;
    impl->speculative_backend  = inputs.speculative_backend;
    impl->proposal_head        = inputs.proposal_head;
    impl->features             = inputs.features;
    impl->use_cuda_graph       = inputs.use_cuda_graph;
    impl->causal_scoring       = inputs.causal_scoring;
    impl->device               = inputs.device;
    impl->tp                  = inputs.tp;
    impl->tp_mailbox          = inputs.tp_mailbox;
    impl->multiprocessor_count = inputs.multiprocessor_count;
    impl->context_cache        = inputs.context_cache;
    impl->kv_storage           = inputs.kv_storage;
    impl->persistent           = persistent_layout(*impl, 0);
    // Rank 1's layout is rank 0's without the masked drafter's state; the drafter runs on rank 0
    // alone, and rank 0's reservation bounds both ranks'.
    if (impl->tp != 1) { impl->peer_persistent = persistent_layout(*impl, 1); }
    if (!impl->context_cache.host_capacity_bytes) {
        // Default Host capacity covers 8 GiB of KV bytes plus eight complete StateImages.
        impl->context_cache.host_capacity_bytes =
            checked_add(8ULL * 1024 * 1024 * 1024,
                        checked_mul(8, impl->persistent.state_images.host.image_bytes,
                                    "Host context state default overflow"),
                        "Host context default overflow");
    }
    impl->workspace = build_workspace_plan(*impl);
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        if (impl->speculative_backend == SpeculativeBackend::None) {
            // Per device; tp 2 uses its measured allowance (see kTp2OrdinaryGraphAllowance).
            const std::size_t per_batch = impl->tp == 1 ? 12ULL * kMiB : kTp2OrdinaryGraphAllowance;
            impl->graph_allowance_bytes =
                checked_mul(per_batch, impl->max_concurrency, "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles = mtp_graph_profiles(impl->capacity, impl->draft_window);
            // Per device; tp 2 uses its measured class allowance (see kTp2MtpGraphClassAllowance).
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) -> std::size_t {
                    if (impl->tp != 1) { return kTp2MtpGraphClassAllowance; }
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + 2ULL * impl->draft_window);
                    return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
        } else {
            const auto profiles = dflash_graph_profiles(impl->speculative_backend, impl->capacity,
                                                        impl->draft_window);
            const auto per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) -> std::size_t {
                    // Per device; tp 2 uses its measured class allowance (see
                    // kTp2DFlash2GraphClassAllowance: 18.0 / 12.0 MiB over the five classes on the
                    // two ranks, 3.6 MiB per class with rank 0 also holding the drafter).
                    if (impl->tp != 1) { return kTp2DFlash2GraphClassAllowance; }
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL);
                    // Long profiles also materialize driver execution storage; that shared
                    // cost does not shrink with the number of graph executables.
                    return (final_visible <= 4096 ? 64ULL : 192ULL) * kMiB;
                },
                "DFlash graph allowance");
            // Forward retains the draft's topology classes; finish has one small executable
            // per exact B, independently of context length. At tp 2 both families' classes are
            // dual-device, so they take the same measured per-rank class allowance.
            const std::size_t finish_allowance =
                impl->tp == 1 ? 8ULL * kMiB : kTp2DFlash2GraphClassAllowance;
            impl->graph_allowance_bytes =
                checked_mul(per_batch_allowance + finish_allowance, impl->max_concurrency,
                            "DFlash exact-b graph allowance");
        }
    }

    // Rank 0's layout bounds rank 1's, which only omits the drafter's state; both ranks are
    // budgeted for it.
    impl->device_reservation_bytes = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options) {
    validate_target_options(parameters, device, options);
    SequencePlanningInputs inputs{
        .parameters           = &parameters,
        .capacity             = options.max_context,
        .max_concurrency      = options.max_concurrency,
        .prefill_chunk        = std::min(options.prefill_chunk, options.max_context),
        .draft_window         = options.speculative.draft_tokens,
        .speculative_backend  = options.speculative.backend,
        .kv_storage           = options.kv_cache,
        .proposal_head        = options.speculative.proposal_head,
        .features             = models::load_options(options),
        .use_cuda_graph       = options.use_cuda_graph,
        .causal_scoring       = options.purpose == EnginePurpose::CausalScoring,
        .device               = options.device,
        .multiprocessor_count = device.multiprocessor_count(),
        .tp                   = options.tp,
        .tp_mailbox           = options.tp_mailbox,
        .context_cache        = options.context_cache,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_5::detail::SequencePlannerImpl>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.5 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.5 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.5 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninfer::models::qwen3_5::detail
