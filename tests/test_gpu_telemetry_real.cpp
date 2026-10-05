// Real NVML on the local GPUs: every CUDA device resolves by UUID and reports a clock, power, a
// temperature and an energy delta. Skips (77) on a host without CUDA devices or NVML. Reads only
// device properties and NVML counters: no CUDA context, no device memory.
#include "serve/gpu_telemetry.h"
#include "serve/request_log.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

int main() {
    using namespace ninfer::serve;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("skip: no CUDA device");
        return 77;
    }
    std::vector<GpuTelemetryDevice> devices;
    for (int device = 0; device < count; ++device) {
        devices.push_back(
            {.device = device, .uuid = query_server_log_environment(device).gpu_uuid});
    }
    GpuTelemetry telemetry(devices);
    if (!telemetry.available()) {
        std::puts("skip: libnvidia-ml.so.1 unavailable");
        return 77;
    }
    int failures        = 0;
    const auto baseline = telemetry.sample();
    if (!baseline || baseline->size() != devices.size()) {
        std::puts("baseline sample does not cover every device");
        return 1;
    }
    for (const GpuTelemetrySample& sample : *baseline) {
        if (sample.energy_joules) {
            std::printf("device %d: the baseline sample reported an energy delta\n", sample.device);
            ++failures;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    const auto started = std::chrono::steady_clock::now();
    const auto samples = telemetry.sample();
    const double sample_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started)
            .count();
    for (const GpuTelemetrySample& sample : *samples) {
        std::printf("device %d: sm %u MHz, mem %u MHz, %.2f W, %.3f J in 1 s, %u C, reasons",
                    sample.device, sample.sm_clock_mhz.value_or(0),
                    sample.memory_clock_mhz.value_or(0), sample.power_watts.value_or(-1.0),
                    sample.energy_joules.value_or(-1.0), sample.temperature_celsius.value_or(0));
        for (const std::string& name :
             clock_event_reason_names(sample.clock_event_reasons.value_or(0))) {
            std::printf(" %s", name.c_str());
        }
        std::puts("");
        if (!sample.sm_clock_mhz || *sample.sm_clock_mhz == 0 || !sample.memory_clock_mhz ||
            !sample.power_watts || *sample.power_watts <= 0.0 || !sample.temperature_celsius ||
            !sample.energy_joules || *sample.energy_joules <= 0.0 || !sample.clock_event_reasons) {
            std::printf("device %d: an NVML value is missing or zero\n", sample.device);
            ++failures;
        }
    }
    std::printf("one sample of %zu devices: %.0f us\n", samples->size(), sample_us);
    if (failures == 0) { std::puts("ok"); }
    return failures == 0 ? 0 : 1;
}
