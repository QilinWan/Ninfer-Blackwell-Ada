#pragma once

#include "core/device.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    // Rank 1's Parameters of a two-device Model; null at tensor-parallel width 1.
    const std::unique_ptr<const models::qwen3_5::execution::Parameters> peer_parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

// Single-device construction on `device` (options.tp must be 1).
[[nodiscard]] ConstructedModel construct_model(EngineOptions& options, DeviceContext& device);
// Construction over every rank of `execution`; execution.tp equals options.tp and rank r runs
// on execution.dev[r]. Width 1 is exactly the DeviceContext overload on execution.primary().
[[nodiscard]] ConstructedModel construct_model(EngineOptions& options,
                                               ExecutionContext& execution);

} // namespace ninfer::runtime
