#pragma once

// GPU clock, power and temperature samples for the request JSONL `throughput` event. The NVIDIA
// Management Library is loaded at run time (dlopen of libnvidia-ml.so.1, installed with the
// driver), so the build needs no NVML headers or link dependency and a host without it simply
// records no samples. Sampling runs on the statistics thread, never on the request path.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GpuTelemetryDevice {
    int device = 0;   // CUDA device id, as in --devices.
    std::string uuid; // "GPU-xxxxxxxx-...", the identity NVML and CUDA share.
};

// One device at the end of a statistics interval. A value is empty when NVML could not read it.
struct GpuTelemetrySample {
    int device = 0;
    std::optional<std::uint32_t> sm_clock_mhz;
    std::optional<std::uint32_t> memory_clock_mhz;
    // NVML board power draw (the driver's own short average), at the end of the interval.
    std::optional<double> power_watts;
    // Energy the board consumed since the previous sample, i.e. over the interval.
    std::optional<double> energy_joules;
    std::optional<std::uint32_t> temperature_celsius;
    // NVML clocks-event (throttle) reason bits active at the end of the interval.
    std::optional<std::uint64_t> clock_event_reasons;
};

// Stable names of the NVML clocks-event reason bits, lowest bit first; an undefined bit is named
// by its hexadecimal value.
[[nodiscard]] std::vector<std::string> clock_event_reason_names(std::uint64_t reasons);

class GpuTelemetry {
public:
    explicit GpuTelemetry(std::vector<GpuTelemetryDevice> devices);
    ~GpuTelemetry();

    GpuTelemetry(const GpuTelemetry&)            = delete;
    GpuTelemetry& operator=(const GpuTelemetry&) = delete;

    // False when libnvidia-ml.so.1 or nvmlInit is unavailable.
    [[nodiscard]] bool available() const noexcept;

    // One sample per configured device, in construction order; nullopt when NVML is unavailable.
    // energy_joules covers the time since the previous call, so the first call only sets the
    // baseline. Not thread-safe: one caller (the statistics thread) owns that baseline.
    [[nodiscard]] std::optional<std::vector<GpuTelemetrySample>> sample();

private:
    struct Library;
    struct Device;
    std::unique_ptr<Library> library_;
    std::vector<Device> devices_;
};

} // namespace ninfer::serve
