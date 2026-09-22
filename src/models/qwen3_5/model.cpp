#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights, DeviceWeights bound,
             int device_count, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), device_count_(device_count),
      resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

std::size_t Model::checked_device(int device) const {
    if (device < 0 || device >= device_count_) {
        throw std::out_of_range("model device " + std::to_string(device) + " is outside the " +
                                std::to_string(device_count_) + "-device backing");
    }
    return static_cast<std::size_t>(device);
}

bool Model::resident(WeightId id, int device) const {
    return !bound_[checked_device(device)].at(id.index).view.parts.empty();
}

const BoundWeight& Model::weight(WeightId id, int device) const {
    const auto& bound = bound_[checked_device(device)].at(id.index);
    if (bound.view.parts.empty()) {
        throw std::invalid_argument(bound.name + " is not resident on device " +
                                    std::to_string(device));
    }
    return bound;
}

ops::WeightInput Model::input(WeightUseId id, int device) const {
    const auto& parameter = weight(id.parameter, device);
    const auto& use       = parameter.uses.at(id.use_index);
    return {parameter.view, use.policy, use.activation_input_divisor};
}

ops::WeightInput Model::input(WeightId id, int device) const {
    if (weight(id, device).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0}, device);
}

} // namespace ninfer::models::qwen3_5
