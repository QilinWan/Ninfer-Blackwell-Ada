#include "runtime/contract/yarn.h"
#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    validate_yarn(options.yarn, options.max_context);
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    if (options.tp == 2) {
        // Rejected before the artifact is read: the two-device schedule covers prefill, the
        // ordinary decode round, the MTP round and the DFlash2 round of the dense Text model only
        // (models/qwen3_5/execution/text.h).
        if (options.purpose != EnginePurpose::Generation) {
            throw std::invalid_argument("Engine tp 2 does not support CausalScoring");
        }
        if (options.speculative.backend == SpeculativeBackend::DFlash) {
            throw std::invalid_argument("Engine tp 2 does not support DFlash speculative decoding");
        }
        // The DFlash2 drafter runs whole on rank 0 and ranks its candidates over one complete
        // proposal head; the full output head is split by vocabulary across the ranks, so only
        // the optimized head, which rank 0 holds whole, can serve it.
        if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
            options.speculative.proposal_head != ProposalHead::Optimized) {
            throw std::invalid_argument(
                "Engine tp 2 DFlash2 requires the optimized proposal head (--lm-head-draft)");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("Engine tp 2 does not support Vision");
        }
        if (options.kv_cache != KvCacheStorage::BFloat16 &&
            options.kv_cache != KvCacheStorage::Int8Group64) {
            throw std::invalid_argument("Engine tp 2 supports only bf16 and int8 KV caches");
        }
        if (options.context_cache.host_capacity_bytes.value_or(0) != 0) {
            throw std::invalid_argument(
                "Engine tp 2 requires a context_cache Host capacity of 0: rank 1's KV and state "
                "have no Host tier");
        }
    }
}

std::size_t free_device_bytes(int device) {
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(device));
    std::size_t free_bytes   = 0;
    std::size_t total_bytes  = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    CUDA_CHECK(cudaSetDevice(previous));
    CUDA_CHECK(status);
    return free_bytes;
}

// The bottleneck rank's free bytes; one rank at width 1.
std::size_t free_runtime_bytes(const DeviceContext& primary, const ExecutionContext* execution) {
    std::size_t free_bytes = free_device_bytes(primary.device);
    if (execution != nullptr) {
        for (int rank = 1; rank < execution->tp; ++rank) {
            free_bytes =
                std::min(free_bytes,
                         free_device_bytes(execution->dev[static_cast<std::size_t>(rank)]->device));
        }
    }
    return free_bytes;
}

void synchronize_ranks(const DeviceContext& primary, const ExecutionContext* execution) {
    if (execution != nullptr) {
        for (int rank = 1; rank < execution->tp; ++rank) {
            execution->dev[static_cast<std::size_t>(rank)]->synchronize();
        }
    }
    primary.synchronize();
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{
                   .enabled = false, .device_state_slots = 0, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    if (options.tp != 1 && options.tp != 2) {
        throw std::invalid_argument("Engine tp must be 1 or 2");
    }
    if (options.devices.empty()) {
        if (options.tp != 1) {
            throw std::invalid_argument("Engine tp 2 requires one device id per rank in devices");
        }
        options.devices = {options.device};
    }
    if (options.devices.size() != static_cast<std::size_t>(options.tp) ||
        options.devices.front() != options.device) {
        throw std::invalid_argument(
            "Engine devices must list one id per tensor-parallel rank, starting with device");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    // At tp 2 the Host tier is off, so every endpoint, rewrite and anchor checkpoint must live in a
    // Device StateImage. An in-prefill capture (the TurnClosure rewrite a follow-up turn resumes
    // from, a long anchor, the default implicit shared prefix) that finds the pool full is skipped
    // instead of snapshotted to Host: once earlier conversations fill the pool, a new conversation
    // keeps only its endpoint and its next turn re-prefills the whole prompt. The tp 2 default pool
    // therefore matches the private catalog, one checkpoint image per retained continuation; each
    // image costs one StateImage per rank (about 73 MiB for Qwen3.8 27B).
    const std::uint64_t default_device_states = options.tp == 2
                                                    ? std::max<std::uint64_t>(2ULL * concurrency,
                                                                            8ULL)
                                                    : static_cast<std::uint64_t>(concurrency);
    cache.device_state_slots =
        cache.device_state_slots.value_or(static_cast<std::uint32_t>(default_device_states));
    if (options.tp == 2) {
        if (cache.host_capacity_bytes && *cache.host_capacity_bytes != 0) {
            throw std::invalid_argument(
                "tensor-parallel Engine requires a Host context capacity of 0");
        }
        cache.host_capacity_bytes = 0;
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model, options.yarn),
      // Rank 1's operand set carries the same immutable YaRN coefficients as rank 0: a
      // tensor-parallel round splits heads, never the rope scaling.
      peer_parameters(model->device_count() == 2
                          ? std::make_unique<const models::qwen3_5::execution::Parameters>(
                                *model, options.yarn, 1)
                          : nullptr),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .grammar_cache_bytes      = options.grammar_cache_bytes,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

namespace {

// `execution` is null at tensor-parallel width 1, where `device` is the only device.
ConstructedModel construct_model_on(EngineOptions& options, DeviceContext& device,
                                    ExecutionContext* execution) {
    validate_options(options);
    if ((execution != nullptr) != (options.tp == 2)) {
        throw std::logic_error("Engine execution context does not match the tensor-parallel width");
    }
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model = execution != nullptr
                     ? models::qwen3_5::materialize_model(std::move(plan), *execution,
                                                          &options.startup_observer)
                     : models::qwen3_5::materialize_model(std::move(plan), device,
                                                          &options.startup_observer);
    synchronize_ranks(device, execution);
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    if ((execution != nullptr) != (instance->peer_parameters != nullptr)) {
        throw std::logic_error("loaded Model device count does not match the Engine tp");
    }
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    // Every rank reserves the same per-rank layout and addresses the same KV pages, so one page
    // count is resolved against the tightest rank's own free memory.
    std::vector<std::size_t> rank_budgets{free_device_bytes(device.device)};
    if (execution != nullptr) {
        for (int rank = 1; rank < execution->tp; ++rank) {
            rank_budgets.push_back(
                free_device_bytes(execution->dev[static_cast<std::size_t>(rank)]->device));
        }
    }
    auto resolution =
        resolve_kv_capacity_symmetric(options.kv_capacity, planner.capacity_curve(), rank_budgets);
    auto sequence = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
    instance->kv_capacity_resolution          = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program =
        execution != nullptr
            ? models::qwen3_5::create_program(instance->parameters, *instance->peer_parameters,
                                              std::move(sequence), *execution,
                                              options.startup_observer)
            : models::qwen3_5::create_program(instance->parameters, std::move(sequence), device,
                                              options.startup_observer);
    synchronize_ranks(device, execution);
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes =
        free_runtime_bytes(device, execution);
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    for (int i = 0; i < stats.device_count; ++i) {
        const auto rank = static_cast<std::size_t>(i);
        summary.devices.push_back(
            {.device         = execution != nullptr ? execution->dev[rank]->device : device.device,
             .capacity_bytes = stats.per_device_capacity_bytes[rank],
             .host_to_device_bytes = stats.per_device_h2d_bytes[rank],
             .sharded_bytes        = stats.sharded_bytes[rank],
             .replicated_bytes     = stats.replicated_bytes[rank],
             .local_bytes          = stats.local_bytes[rank]});
    }
    summary.context_cost         = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    if (options.tp != 1) {
        throw std::invalid_argument("single-device Engine construction requires tp 1");
    }
    return construct_model_on(options, device, nullptr);
}

ConstructedModel construct_model(EngineOptions& options, ExecutionContext& execution) {
    if (execution.tp != options.tp || !execution.dev[0]) {
        throw std::invalid_argument("Engine execution context does not match options.tp");
    }
    return construct_model_on(options, execution.primary(),
                              execution.tp == 1 ? nullptr : &execution);
}

} // namespace ninfer::runtime
