#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    // Tensor-parallel degree of the backing: 1, or 2 with one view set per rank.
    [[nodiscard]] int device_count() const noexcept { return device_count_; }

    // False when `device` holds none of the parameter's parents (a PrimaryOnly or SingleDevice
    // placement on the other rank).
    [[nodiscard]] bool resident(WeightId id, int device = 0) const;

    // The view on `device`: a complete parent, or that rank's Rows/Columns shard with the split
    // axis of its logical shape narrowed. Throws when the parameter is not resident there.
    [[nodiscard]] const BoundWeight& weight(WeightId id, int device = 0) const;

    [[nodiscard]] ops::WeightInput input(WeightUseId id, int device = 0) const;
    [[nodiscard]] ops::WeightInput input(WeightId id, int device = 0) const;

    // Indexed by WeightId; entries not resident on `device` have an empty view.
    [[nodiscard]] std::span<const BoundWeight> weight_data(int device = 0) const {
        return bound_.at(checked_device(device));
    }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, ExecutionContext&,
                                                    const StartupObserver*);
    using DeviceWeights = std::array<std::vector<BoundWeight>, artifact::kMaximumDevices>;

    Model(Config config, LoadOptions options, ModelWeights weights, DeviceWeights bound,
          int device_count, FrontendResources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing);

    [[nodiscard]] std::size_t checked_device(int device) const;

    // Destroy all borrowers before backing. The caller keeps its DeviceContext or ExecutionContext
    // alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    DeviceWeights bound_;
    int device_count_ = 1;
    FrontendResources resources_;
    InstanceInfo info_;
};

} // namespace ninfer::models::qwen3_5
