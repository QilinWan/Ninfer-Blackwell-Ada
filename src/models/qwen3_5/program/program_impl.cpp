#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/startup.h"
#include "core/device.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

namespace {

// Both ranks of a tensor-parallel Program start from zeroed persistent and workspace arenas, so
// the two ranks' KV pages, StateImages and scratch hold identical bytes before their first
// write. A single-device Program's arenas start undefined, as every owner fills what it reads.
ZeroFill tensor_parallel_zero_fill(const SequencePlanImpl& plan) noexcept {
    return plan.peer_persistent ? ZeroFill::Yes : ZeroFill::No;
}

} // namespace

const PersistentLayout& ProgramImpl::PeerRuntime::layout(const SequencePlanImpl& plan) {
    if (!plan.peer_persistent || plan.peer_persistent->dflash ||
        plan.peer_persistent->state_images.dflash_local) {
        throw std::logic_error("tensor-parallel plan has no rank 1 persistent layout");
    }
    return *plan.peer_persistent;
}

ProgramImpl::PeerRuntime::PeerRuntime(DeviceContext& peer_device, const SequencePlanImpl& plan)
    : device(peer_device), persistent(layout(plan).bytes, ZeroFill::Yes),
      workspace_storage(plan.workspace.capacity, ZeroFill::Yes),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}) {
    const PersistentLayout& own = layout(plan);
    const DeviceSpan backing    = persistent.alloc_bytes(own.bytes, 256);
    decoder                     = std::make_unique<qwen3_5::DecoderState>(backing, own.decoder);
    state_images = std::make_unique<qwen3_5::StateImageDevicePool>(backing, own.state_images);
    if (own.replay_records) {
        replay_records.emplace(backing, *own.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    io             = qwen3_5::RoundState(backing, own.round);
    prefill_hidden = own.prefill_hidden.bind(backing);
    if (static_cast<int>(io.ordinary.has_value()) + static_cast<int>(io.mtp_decode.has_value()) +
            static_cast<int>(io.dflash_decode.has_value()) !=
        1) {
        throw std::logic_error("tensor-parallel rank 1 needs exactly one of the ordinary, MTP and "
                               "DFlash decode frames");
    }
    if (io.ordinary) { ordinary = execution::ordinary_peer_frame(*io.ordinary); }
}

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer,
                         ExecutionContext* execution_in,
                         const execution::Parameters* peer_parameters_in)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      kv_storage(plan.kv_storage), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_cuda_graph(plan.use_cuda_graph),
      causal_scoring(plan.causal_scoring), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes, tensor_parallel_zero_fill(plan)),
      workspace_storage(plan.workspace.capacity, tensor_parallel_zero_fill(plan)),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      round_host(plan.causal_scoring
                     ? std::nullopt
                     : std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::PrefillRoundHost))),
      score_logprobs_host(plan.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                    kCausalScoreTile * sizeof(float))
                                              : std::nullopt),
      ordinary_host(
          !plan.causal_scoring && plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_5::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::MtpDecodeIngress) +
                                                          sizeof(qwen3_5::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(is_masked_draft_backend(plan.speculative_backend)
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress) +
                                                             sizeof(qwen3_5::DFlashPrefillIngress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)},
      prefill_gpu_timer_(device_in) {
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    if (plan.tp != 1 && plan.tp != execution::kTensorParallelWidth) {
        throw std::invalid_argument("Qwen3.5 Program tensor-parallel width must be 1 or 2");
    }
    if ((plan.tp == execution::kTensorParallelWidth) !=
            (execution_in != nullptr && peer_parameters_in != nullptr) ||
        (execution_in != nullptr && execution_in->tp != plan.tp)) {
        throw std::invalid_argument(
            "Qwen3.5 Program execution context does not match the sequence plan width");
    }
    if (execution_in != nullptr) {
        if (!execution_in->dev[0] || !execution_in->dev[1] || &*execution_in->dev[0] != &device ||
            &peer_parameters_in->model != &parameters.model || peer_parameters_in->device != 1 ||
            parameters.device != 0 || parameters.model.device_count() != plan.tp) {
            throw std::invalid_argument(
                "tensor-parallel Program requires rank 0 and rank 1 Parameters of one two-device "
                "Model on its ExecutionContext");
        }
        // Rank 1's KV pages and StateImages are mirrors of rank 0's and have no Host replica, so
        // the Host tier cannot hold a complete checkpoint. Our ContextCacheOptions exposes one
        // Host budget (host_capacity_bytes) covering StateImages, KV pages and snapshots.
        if (plan.context_cache.host_capacity_bytes.value_or(0) != 0) {
            throw std::invalid_argument(
                "tensor-parallel Program requires a Host context capacity of 0");
        }
        if (speculative_backend == SpeculativeBackend::DFlash || vision_enabled || causal_scoring) {
            throw std::invalid_argument("tensor-parallel Program supports ordinary, MTP and "
                                        "DFlash2 text generation only");
        }
        execution_context = execution_in;
        peer_parameters   = peer_parameters_in;
        {
            // cudaMalloc is neither stream-ordered nor capturable: rank 1's storage is allocated
            // once, here, with rank 1 current.
            const ScopedCurrentDevice rank1(execution_context->dev[1]->device);
            peer = std::make_unique<PeerRuntime>(*execution_context->dev[1], plan);
            peer_events.emplace(*execution_context);
            if (use_cuda_graph) {
                graph_bridge.emplace(execution_context->dev[0]->device,
                                     execution_context->dev[1]->device);
                if (plan.tp_mailbox) {
                    // One slot carries the widest single-request exchange, the [hidden, K+1]
                    // verification activation; wider (multi-request) payloads stay staged.
                    const std::size_t slot_bytes =
                        static_cast<std::size_t>(parameters.model.config().text.hidden_size) *
                        (static_cast<std::size_t>(draft_window) + 1U) * sizeof(std::uint16_t);
                    peer_mailbox.emplace(*execution_context, slot_bytes);
                    peer_events->attach_mailbox(&*peer_mailbox);
                    probe_peer_mailbox();
                } else {
                    tp_transport_status.transport = "copies";
                }
            } else {
                tp_transport_status.transport = "copies";
            }
        }
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder      = std::make_unique<qwen3_5::DecoderState>(backing, plan.persistent.decoder);
    state_images = std::make_unique<StateImageDevicePool>(backing, plan.persistent.state_images);
    if (peer) { attach_tensor_parallel_mirrors(); }
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    if (plan.persistent.dflash) {
        auto* local = state_images->dflash_local();
        if (!local) { throw std::logic_error("DFlash StateImage has no local state"); }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    const auto host_bytes = plan.context_cache.host_capacity_bytes.value();
    std::vector<HostKVPageLayout> layouts{
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry())};
    text_host_kv_page_stride = layouts.front().page_stride;
    if (const auto* backend = backend_kv_cache()) {
        auto layout                 = plan_host_kv_page_layout(backend->page_pool().geometry());
        backend_host_kv_page_stride = layout.page_stride;
        if (layout != layouts.front()) { layouts.push_back(std::move(layout)); }
    }
    std::size_t minimum_stride    = state_images->host_layout().image_bytes;
    std::size_t minimum_kv_stride = layouts.front().page_stride;
    for (const auto& layout : layouts) {
        minimum_stride    = std::min(minimum_stride, layout.page_stride);
        minimum_kv_stride = std::min(minimum_kv_stride, layout.page_stride);
    }
    const auto checked_count = [](std::uint64_t value) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("context resource descriptor capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(value);
    };
    const auto logical_states = checked_count(state_images->slot_count() +
                                              host_bytes / state_images->host_layout().image_bytes);
    // One immutable image can serve private replay, an explicit anchor, a public view and an
    // exact-hit endpoint. Alias descriptors are bounded separately from physical state slots.
    checkpoints.resize(checked_count(4ULL * logical_states + 2ULL * max_concurrency));
    const auto address_capacity = checked_count(checkpoints.size() + max_concurrency + 2U);
    if (host_bytes) {
        StartupPhaseScope phase(startup_observer, StartupPhase::HostContextPin,
                                StartupProgressUnit::Bytes, host_bytes);
        host_context_arena = std::make_unique<HostContextArena>(host_bytes, minimum_stride);
        host_state_images =
            std::make_unique<HostStatePool>(*host_context_arena, state_images->host_layout());
        host_kv_arena      = std::make_unique<HostKVArena>(*host_context_arena, layouts);
        const auto extents = checked_count(host_bytes / minimum_kv_stride);
        if (extents) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(*host_kv_arena, extents);
        }
        phase.complete(host_bytes, host_bytes);
    }
    state_store =
        std::make_unique<StateImageStore>(*state_images, host_state_images.get(), logical_states);
    const auto logical_pages = [&](const DeviceKVPagePool& pool) {
        const auto stride = plan_host_kv_page_layout(pool.geometry()).page_stride;
        return checked_count(pool.capacity_pages() + host_bytes / stride);
    };
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_pages(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    if (auto* backend = backend_kv_cache()) {
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_pages(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }

    io = qwen3_5::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() !=
        (!causal_scoring && speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    if (plan.persistent.grammar_masks) {
        grammar_masks_device = plan.persistent.grammar_masks->bind(backing);
        grammar_masks_host.emplace(grammar_masks_device.bytes());
    }
    if (is_masked_draft_backend(speculative_backend)) {
        dflash_draft_handoff.emplace(device, draft_window * max_concurrency);
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        lane_epochs[lane]    = 1;
        sequences[lane].lane = lane;
    }
    if (peer) {
        // Rank 1's prefill row scalar starts at 0; every publish that writes rank 0's writes it.
        set_peer_i32(peer->io.text_kv_table_row, 0);
        set_peer_i32(peer->io.backend_kv_table_row, 0);
        const bool mtp         = speculative_backend == SpeculativeBackend::Mtp;
        const bool speculative = speculative_backend != SpeculativeBackend::None;
        if (mtp && (!peer->io.mtp || !peer->io.mtp_decode || !peer->replay_records ||
                    peer->decoder->mtp_cache() == nullptr)) {
            throw std::logic_error("tensor-parallel rank 1 has no MTP storage");
        }
        // Rank 1 verifies DFlash2 proposals in its own copy of the decode frame and records its
        // GDN heads for the commit fold; it holds none of the drafter's state.
        if (is_masked_draft_backend(speculative_backend) &&
            (!peer->io.dflash_decode || !peer->replay_records ||
             peer->state_images->dflash_local() != nullptr)) {
            throw std::logic_error("tensor-parallel rank 1 has no DFlash2 verification storage");
        }
        tp_execution.emplace(execution::TpExecution{
            .execution            = execution_context,
            .events               = &*peer_events,
            .parameters           = peer_parameters,
            .work                 = &peer->work,
            .linear_attention     = &peer->state_images->linear(),
            .text_cache           = &peer->decoder->text_kv,
            .text_kv_table_row    = peer->io.text_kv_table_row,
            .ordinary             = peer->io.ordinary ? &peer->ordinary : nullptr,
            .replay_records       = speculative ? &*peer->replay_records : nullptr,
            .mtp_cache            = mtp ? peer->decoder->mtp_cache() : nullptr,
            .mtp_kv               = {},
            .backend_kv_table_row = mtp ? peer->io.backend_kv_table_row : Tensor{},
            .mtp                  = mtp ? &*peer->io.mtp : nullptr,
            .prefill_hidden       = mtp ? peer->prefill_hidden : Tensor{},
        });
        const char* draft_env = std::getenv("NINFER_TP_MAILBOX_DRAFT");
        if (mtp && peer_mailbox && draft_env != nullptr && std::string_view(draft_env) == "copies") {
            tp_execution->staged_draft_collectives = true;
            tp_transport_status.transport          = "mailbox, MTP draft on copies";
        }
    }
    host_tokens = round_host
                      ? &static_cast<qwen3_5::PrefillRoundHost*>(round_host->data())->sampled_token
                      : nullptr;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_5::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_5::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_5::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_5::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_5::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_5::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_5::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_5::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress));
        *dflash_host_ingress        = {};
        *dflash_host_egress         = {};
        dflash_prefill_host_ingress = reinterpret_cast<qwen3_5::DFlashPrefillIngress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress) + sizeof(qwen3_5::DFlashDecodeEgress));
        *dflash_prefill_host_ingress = {};
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    }
    synchronize_devices();
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        prepare_graphs();
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    if (peer) { peer->work.reset(); }
    workspace_logical_peak_bytes = 0;
}

ProgramImpl::~ProgramImpl() noexcept {
    if (peer) {
        if (peer->device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(peer->device.transfer_stream);
        }
        if (peer->device.stream != nullptr) { (void)cudaStreamSynchronize(peer->device.stream); }
    }
    if (device.transfer_stream != nullptr) { (void)cudaStreamSynchronize(device.transfer_stream); }
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

void ProgramImpl::attach_tensor_parallel_mirrors() {
    // Rank 0's pools own all page, row and slot bookkeeping; every physical mutation they issue
    // (page zero/copy, execution-row acquire/release/publication, StateImage zero/copy) is
    // replayed at the same indices on rank 1's pool, on rank 1's stream. Attached before any
    // mutation, so both ranks start from the same, empty addressing.
    const DeviceContext& rank1 = peer->device;
    const DeviceKVMirror where{.device = rank1.device, .stream = rank1.stream};
    decoder->text_kv.page_pool().attach_mirror(peer->decoder->text_kv.page_pool(), where);
    decoder->text_kv.execution_tables().attach_mirror(peer->decoder->text_kv.execution_tables(),
                                                      where);
    if (qwen3_5::PagedKVCache* mtp = decoder->mtp_cache()) {
        // The MTP rows are leased with the sequence's backend address space; the mirror takes and
        // releases rank 1's lease on the same row with rank 0's.
        qwen3_5::PagedKVCache* peer_mtp = peer->decoder->mtp_cache();
        if (peer_mtp == nullptr) { throw std::logic_error("tensor-parallel rank 1 has no MTP KV"); }
        mtp->page_pool().attach_mirror(peer_mtp->page_pool(), where);
        mtp->execution_tables().attach_mirror(peer_mtp->execution_tables(), where);
    }
    state_images->attach_mirror(*peer->state_images, rank1.device, rank1.stream);
}

std::optional<execution::TpExecution>
ProgramImpl::prefill_tp_binding(const SequenceState& sequence) const {
    if (!tp_execution) { return std::nullopt; }
    execution::TpExecution binding = *tp_execution;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend ||
            !backend_kv_addresses->active(*sequence.kv->backend)) {
            throw std::logic_error("sequence has no active MTP KV execution mapping");
        }
        const KVExecutionRowLease& row = decoder->mtp_cache()->execution_tables().mirror_row(
            backend_kv_addresses->execution_row(*sequence.kv->backend).handle());
        binding.mtp_kv = peer->decoder->mtp_cache()->execution_view(row);
    }
    return binding;
}

void ProgramImpl::probe_peer_mailbox() {
    const char* setting_env = std::getenv("NINFER_TP_MAILBOX_PROBE");
    const std::string_view setting = setting_env != nullptr ? setting_env : "";
    if (setting == "off") {
        tp_transport_status.transport = "mailbox";
        return;
    }
    // A slow-but-alive mailbox is worse than the copies: every decode step crosses it about
    // twice per layer, so a round trip past this bound would cost more than the staged path.
    constexpr double kMaxRoundTripMs = 50.0;
    constexpr std::size_t kBytes     = 4096; // whole 16-byte vectors, inside every slot
    constexpr std::int32_t kElements = static_cast<std::int32_t>(kBytes / sizeof(std::uint16_t));
    DeviceContext& rank0 = *execution_context->dev[0];
    DeviceContext& rank1 = *execution_context->dev[1];

    struct Scratch {
        int device       = 0;
        void* buffer     = nullptr;
        void* staging    = nullptr;
        explicit Scratch(int device_in) : device(device_in) {
            const ScopedCurrentDevice current(device);
            CUDA_CHECK(cudaMalloc(&buffer, kBytes));
            CUDA_CHECK(cudaMalloc(&staging, kBytes));
            CUDA_CHECK(cudaMemset(buffer, 0, kBytes));
            CUDA_CHECK(cudaMemset(staging, 0, kBytes));
        }
        ~Scratch() {
            const ScopedCurrentDevice current(device);
            cudaFree(buffer);
            cudaFree(staging);
        }
        Scratch(const Scratch&)            = delete;
        Scratch& operator=(const Scratch&) = delete;
    };

    bool hang         = false;
    bool selected     = true;
    double elapsed_ms = 0.0;
    {
        const Scratch scratch0(rank0.device);
        const Scratch scratch1(rank1.device);
        const std::array<Tensor, 2> buffer{Tensor(scratch0.buffer, DType::BF16, {1, kElements}),
                                           Tensor(scratch1.buffer, DType::BF16, {1, kElements})};
        const std::array<Tensor, 2> staging{
            Tensor(scratch0.staging, DType::BF16, {1, kElements}),
            Tensor(scratch1.staging, DType::BF16, {1, kElements})};
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        {
            const ScopedCurrentDevice current(rank0.device);
            CUDA_CHECK(cudaDeviceSynchronize());
            {
                const ScopedCurrentDevice peer_sync(rank1.device);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            definition.capture(
                rank0.stream,
                [&] {
                    if (setting == "fail") {
                        // Only rank 0 enqueues its half: the poller must give up and report.
                        peer_mailbox->enqueue_exchange_sum(0, peer_mailbox->take_capture_slot(),
                                                           scratch0.buffer, kBytes, rank0.stream);
                    } else {
                        ops::allreduce_sum(buffer, staging, *execution_context, *peer_events);
                    }
                    CUDA_CHECK(cudaSetDevice(rank0.device));
                },
                DecodeGraphPeerCapture{.bridge = &*graph_bridge, .stream = rank1.stream});
            // The mailbox exchange is one kernel node per device; the staged path is more.
            selected = definition.node_count() == 2;
            executable.instantiate(definition);
            const auto start = std::chrono::steady_clock::now();
            executable.launch(rank0.stream);
            CUDA_CHECK(cudaStreamSynchronize(rank0.stream));
            {
                const ScopedCurrentDevice peer_sync(rank1.device);
                CUDA_CHECK(cudaStreamSynchronize(rank1.stream));
            }
            elapsed_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
        }
        hang = peer_mailbox->hang_reported();
    }
    tp_transport_status.probe_ms = elapsed_ms;
    if (!hang && selected && elapsed_ms <= kMaxRoundTripMs) {
        tp_transport_status.transport = "mailbox";
        return;
    }
    if (hang) {
        tp_transport_status.fallback = "the startup probe exchange timed out";
    } else if (!selected) {
        tp_transport_status.fallback = "the startup probe did not capture a mailbox exchange";
    } else {
        tp_transport_status.fallback = "the startup probe exchange took " +
                                       std::to_string(static_cast<long>(elapsed_ms)) + " ms";
    }
    tp_transport_status.transport = "copies";
    peer_events->attach_mailbox(nullptr);
    peer_mailbox.reset();
}

void ProgramImpl::synchronize_devices() const {
    if (peer) { peer->device.synchronize(); }
    device.synchronize();
    if (peer_mailbox && peer_mailbox->hang_reported()) {
        std::string transport = tp_transport_status.transport;
        if (tp_transport_status.probe_ms > 0.0) {
            transport += ", startup probe " +
                         std::to_string(static_cast<long>(tp_transport_status.probe_ms * 1000.0)) +
                         " us";
        }
        throw std::runtime_error("tensor-parallel mailbox exchange timed out waiting for the peer "
                                 "device (captured all-reduces: " +
                                 transport + "); the two ranks' results diverged");
    }
}

bool ProgramImpl::degrade_peer_mailbox() {
    if (!peer_mailbox || !tp_execution) { return false; }
    // The executables bake in the hung mailbox's slab and words: they go first. This tree records
    // the speculative round as one forward and one finish family; both are recaptured with the
    // ordinary family on the stepped-down transport.
    ordinary_graphs            = {};
    speculative_forward_graphs = {};
    speculative_finish_graphs  = {};
    const std::size_t slot_bytes = peer_mailbox->slot_bytes();
    peer_events->attach_mailbox(nullptr);
    peer_mailbox.reset();
    const std::string previous = tp_transport_status.transport;
    const std::string reason   = "an exchange timed out in the first launch of a CUDA graph (" +
                               previous + ")";
    tp_transport_status.fallback +=
        (tp_transport_status.fallback.empty() ? "" : "; then ") + reason;
    if (speculative_backend == SpeculativeBackend::Mtp &&
        !tp_execution->staged_draft_collectives) {
        tp_execution->staged_draft_collectives = true;
        peer_mailbox.emplace(*execution_context, slot_bytes);
        peer_events->attach_mailbox(&*peer_mailbox);
        tp_transport_status.transport = "mailbox, MTP draft on copies";
        return true;
    }
    tp_execution->staged_draft_collectives = false;
    tp_transport_status.transport          = "copies";
    return true;
}

void ProgramImpl::set_peer_i32(Tensor& tensor, std::int32_t value) {
    if (!peer) { throw std::logic_error("tensor-parallel rank 1 is unavailable"); }
    const ScopedCurrentDevice rank1(peer->device.device);
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice,
                               peer->device.stream));
}

void ProgramImpl::publish_peer_kv_rows(const SequenceState& sequence) {
    if (!peer) { return; }
    // Rank 1's prefill reads its own row scalars. Its execution tables are rank 0's mirrors, so the
    // row indices are the same; they are written whenever rank 0's are.
    const std::int32_t row = text_kv_addresses->bound_row(sequence.kv->text);
    const KVExecutionRowLease& peer_row = decoder->text_kv.execution_tables().mirror_row(
        text_kv_addresses->execution_row(sequence.kv->text).handle());
    if (peer_row.row_index() != row) {
        throw std::logic_error("tensor-parallel KV execution rows disagree across ranks");
    }
    set_peer_i32(peer->io.text_kv_table_row, row);

    // The MTP round's backend (MTP KV) row follows the Text row on both ranks, so a speculative
    // prefill, a forced-token prefill and a replay chunk all name the same row on rank 1.
    const std::int32_t backend_row =
        sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
    if (sequence.kv->backend) {
        const qwen3_5::PagedKVCache* backend = backend_kv_cache();
        if (backend == nullptr || backend->execution_tables()
                                       .mirror_row(backend_kv_addresses
                                                       ->execution_row(*sequence.kv->backend)
                                                       .handle())
                                       .row_index() != backend_row) {
            throw std::logic_error(
                "tensor-parallel backend KV execution rows disagree across ranks");
        }
    }
    set_peer_i32(peer->io.backend_kv_table_row, backend_row);
}

std::vector<float> ProgramImpl::causal_score(PreparedPromptData&& prompt,
                                             std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t required_pages         = kv_pages_for_frontier(predictor_count);
    if (required_pages == 0) { throw std::logic_error("causal score requires KV pages"); }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    std::vector<float> output;
    output.reserve(token_count_size - first_target);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;

    try {
        state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(required_pages, 0, device.stream);
        if (!address) { throw std::bad_alloc(); }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, device.stream);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(
                DType::BF16, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids = work.alloc(DType::I32, {columns});
            Tensor logprobs   = work.alloc(DType::FP32, {columns});
            Tensor hidden     = score_hidden->slice(1, 0, columns);
            execution::project(hidden, parameters.text.output_head, logits, work, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                                    cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids,
                                              dimension(parameters.model.resources().public_token_count),
                                              logprobs, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                                    cudaMemcpyDeviceToHost, device.stream));
            device.synchronize();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);

            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = kCausalScoreTile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == kCausalScoreTile) { flush(); }
            }
        }
        flush();
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

void ProgramImpl::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImpl::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

runtime::ContextTransferObservation ProgramImpl::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    out.device          = device.device;
    out.max_context     = capacity;
    out.kv_capacity     = kv_capacity;
    out.kv_cache        = kv_storage;
    const auto& weights = parameters.model.storage_stats();
    // The summary describes rank 0's device; at tp 2 the model's total spans both ranks.
    const std::size_t weight_bytes =
        peer ? weights.per_device_capacity_bytes[0] : weights.device_capacity_bytes;
    out.weights = ArenaMemorySummary{weight_bytes, weight_bytes, weight_bytes};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
        }
    }
    std::size_t active_workspace_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        active_workspace_bytes =
            std::max(active_workspace_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (workspace_plan.vision) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = workspace_plan.vision->max_merged_tokens,
            .general_capacity_bytes = workspace_plan.vision->general_capacity_bytes,
            .encode_peak_bytes      = workspace_plan.vision->encode_peak_bytes,
            .handoff_offset_bytes   = workspace_plan.vision->handoff_offset_bytes,
            .handoff_capacity_bytes = workspace_plan.vision->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
        };
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_observed_bytes    = graph_observed_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    if (host_state_images) { out.host_state_occupied_slots = host_state_images->occupied(); }
    if (host_kv_arena) { out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes(); }
    if (host_context_arena) {
        out.host_context_capacity_bytes = host_context_arena->capacity_bytes();
        out.host_context_occupied_bytes = host_context_arena->occupied_bytes();
        out.host_context_reserved_bytes = host_context_arena->reserved_bytes();
    }
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes    = active_handoff_bytes;
    workspace_logical_peak_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes =
            std::max(workspace_logical_peak_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
}


} // namespace ninfer::models::qwen3_5::detail
