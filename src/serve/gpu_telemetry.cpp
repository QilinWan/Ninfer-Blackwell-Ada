#include "serve/gpu_telemetry.h"

#include <dlfcn.h>

#include <cstdio>
#include <iterator>
#include <utility>

namespace ninfer::serve {
namespace {

// The subset of the NVML C ABI used here (nvml.h, stable since the r3xx drivers). Enumerations
// are passed as int, the C ABI of an enum argument.
using NvmlReturn                     = int;
using NvmlDevice                     = struct NvmlDeviceOpaque*;
constexpr NvmlReturn kNvmlSuccess    = 0;
constexpr int kNvmlClockSm           = 1; // NVML_CLOCK_SM
constexpr int kNvmlClockMemory       = 2; // NVML_CLOCK_MEM
constexpr int kNvmlTemperatureGpu    = 0; // NVML_TEMPERATURE_GPU
constexpr const char* kNvmlLibrary   = "libnvidia-ml.so.1";
constexpr const char* kReasonNames[] = {
    "gpu_idle",                    // 0x1
    "applications_clocks_setting", // 0x2: application or locked clocks
    "sw_power_cap",                // 0x4
    "hw_slowdown",                 // 0x8
    "sync_boost",                  // 0x10
    "sw_thermal_slowdown",         // 0x20
    "hw_thermal_slowdown",         // 0x40
    "hw_power_brake_slowdown",     // 0x80
    "display_clock_setting",       // 0x100
};

using InitFn         = NvmlReturn (*)();
using ShutdownFn     = NvmlReturn (*)();
using HandleByUuidFn = NvmlReturn (*)(const char*, NvmlDevice*);
using ClockInfoFn    = NvmlReturn (*)(NvmlDevice, int, unsigned int*);
using PowerUsageFn   = NvmlReturn (*)(NvmlDevice, unsigned int*);
using TotalEnergyFn  = NvmlReturn (*)(NvmlDevice, unsigned long long*);
using TemperatureFn  = NvmlReturn (*)(NvmlDevice, int, unsigned int*);
using EventReasonsFn = NvmlReturn (*)(NvmlDevice, unsigned long long*);

template <class Function>
Function symbol(void* handle, const char* name) {
    return reinterpret_cast<Function>(::dlsym(handle, name));
}

} // namespace

struct GpuTelemetry::Library {
    void* handle                  = nullptr;
    bool initialized              = false;
    ShutdownFn shutdown           = nullptr;
    HandleByUuidFn handle_by_uuid = nullptr;
    ClockInfoFn clock_info        = nullptr;
    PowerUsageFn power_usage      = nullptr;
    TotalEnergyFn total_energy    = nullptr;
    TemperatureFn temperature     = nullptr;
    EventReasonsFn event_reasons  = nullptr;

    Library() {
        handle = ::dlopen(kNvmlLibrary, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) { return; }
        const auto init = symbol<InitFn>(handle, "nvmlInit_v2");
        shutdown        = symbol<ShutdownFn>(handle, "nvmlShutdown");
        handle_by_uuid  = symbol<HandleByUuidFn>(handle, "nvmlDeviceGetHandleByUUID");
        if (init == nullptr || shutdown == nullptr || handle_by_uuid == nullptr ||
            init() != kNvmlSuccess) {
            return;
        }
        initialized  = true;
        clock_info   = symbol<ClockInfoFn>(handle, "nvmlDeviceGetClockInfo");
        power_usage  = symbol<PowerUsageFn>(handle, "nvmlDeviceGetPowerUsage");
        total_energy = symbol<TotalEnergyFn>(handle, "nvmlDeviceGetTotalEnergyConsumption");
        temperature  = symbol<TemperatureFn>(handle, "nvmlDeviceGetTemperature");
        // Renamed from "throttle" to "clocks event" reasons in the r535 API; same bits.
        event_reasons = symbol<EventReasonsFn>(handle, "nvmlDeviceGetCurrentClocksEventReasons");
        if (event_reasons == nullptr) {
            event_reasons =
                symbol<EventReasonsFn>(handle, "nvmlDeviceGetCurrentClocksThrottleReasons");
        }
    }

    ~Library() {
        if (initialized) { (void)shutdown(); }
        if (handle != nullptr) { (void)::dlclose(handle); }
    }

    Library(const Library&)            = delete;
    Library& operator=(const Library&) = delete;
};

struct GpuTelemetry::Device {
    int device        = 0;
    NvmlDevice handle = nullptr;
    std::optional<unsigned long long> previous_energy_mj;
};

std::vector<std::string> clock_event_reason_names(std::uint64_t reasons) {
    std::vector<std::string> names;
    for (unsigned bit = 0; bit < 64; ++bit) {
        const std::uint64_t mask = std::uint64_t{1} << bit;
        if ((reasons & mask) == 0) { continue; }
        if (bit < std::size(kReasonNames)) {
            names.emplace_back(kReasonNames[bit]);
        } else {
            char name[24];
            std::snprintf(name, sizeof(name), "0x%llx", static_cast<unsigned long long>(mask));
            names.emplace_back(name);
        }
    }
    return names;
}

GpuTelemetry::GpuTelemetry(std::vector<GpuTelemetryDevice> devices)
    : library_(std::make_unique<Library>()) {
    devices_.reserve(devices.size());
    for (const GpuTelemetryDevice& identity : devices) {
        Device device{.device = identity.device};
        if (library_->initialized && !identity.uuid.empty() &&
            library_->handle_by_uuid(identity.uuid.c_str(), &device.handle) != kNvmlSuccess) {
            device.handle = nullptr;
        }
        devices_.push_back(device);
    }
}

GpuTelemetry::~GpuTelemetry() = default;

bool GpuTelemetry::available() const noexcept { return library_->initialized; }

std::optional<std::vector<GpuTelemetrySample>> GpuTelemetry::sample() {
    if (!library_->initialized) { return std::nullopt; }
    const Library& nvml = *library_;
    std::vector<GpuTelemetrySample> samples;
    samples.reserve(devices_.size());
    for (Device& device : devices_) {
        GpuTelemetrySample sample{.device = device.device};
        if (device.handle != nullptr) {
            unsigned int value = 0;
            if (nvml.clock_info != nullptr &&
                nvml.clock_info(device.handle, kNvmlClockSm, &value) == kNvmlSuccess) {
                sample.sm_clock_mhz = value;
            }
            if (nvml.clock_info != nullptr &&
                nvml.clock_info(device.handle, kNvmlClockMemory, &value) == kNvmlSuccess) {
                sample.memory_clock_mhz = value;
            }
            if (nvml.power_usage != nullptr &&
                nvml.power_usage(device.handle, &value) == kNvmlSuccess) {
                sample.power_watts = static_cast<double>(value) * 1.0e-3;
            }
            if (nvml.temperature != nullptr &&
                nvml.temperature(device.handle, kNvmlTemperatureGpu, &value) == kNvmlSuccess) {
                sample.temperature_celsius = value;
            }
            unsigned long long counter = 0;
            if (nvml.event_reasons != nullptr &&
                nvml.event_reasons(device.handle, &counter) == kNvmlSuccess) {
                sample.clock_event_reasons = counter;
            }
            std::optional<unsigned long long> energy;
            if (nvml.total_energy != nullptr &&
                nvml.total_energy(device.handle, &counter) == kNvmlSuccess) {
                energy = counter;
            }
            if (energy && device.previous_energy_mj && *energy >= *device.previous_energy_mj) {
                sample.energy_joules =
                    static_cast<double>(*energy - *device.previous_energy_mj) * 1.0e-3;
            }
            device.previous_energy_mj = energy;
        }
        samples.push_back(sample);
    }
    return samples;
}

} // namespace ninfer::serve
