#pragma once

#include "core/device.h" // CUDA_CHECK

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <mutex>

namespace ninfer::ops::detail {

// Once-per-device guard for one cudaFuncSetAttribute call site.
//
// Kernel function attributes such as cudaFuncAttributeMaxDynamicSharedMemorySize are a
// per-device property. A bare function-local `static const cudaError_t attr =
// cudaFuncSetAttribute(...)` applies the opt-in only on whichever device is current the first time
// the launcher runs; every other device that later launches the same kernel keeps the default
// 48 KiB limit and fails the launch with cudaErrorInvalidValue. The guard replaces that static at
// the same place: declare it as a function-local static next to the launch, so there is one guard
// per enclosing template instantiation (hence per kernel) exactly as before, and call ensure() on
// every launch.
//
// ensure() costs one cudaGetDevice and one acquire load once the current device has been
// configured. The attribute is applied under a mutex the first time the call site runs on each
// device, so a concurrent caller on the same device returns only after the attribute is set.
class FuncAttrPerDevice {
public:
    template <typename Func>
    void ensure(Func* func, cudaFuncAttribute attr, int value) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        if (device >= 0 && device < kMaxDevices &&
            applied_[device].load(std::memory_order_acquire)) {
            return;
        }
        apply(reinterpret_cast<const void*>(func), device, attr, value);
    }

private:
    static constexpr int kMaxDevices = 64;

    void apply(const void* func, int device, cudaFuncAttribute attr, int value) {
        const std::lock_guard<std::mutex> lock(mutex_);
        const bool tracked = device >= 0 && device < kMaxDevices;
        if (tracked && applied_[device].load(std::memory_order_relaxed)) { return; }
        CUDA_CHECK(cudaFuncSetAttribute(func, attr, value));
        // Device ids beyond the table are applied on every call: correct, only slower.
        if (tracked) { applied_[device].store(true, std::memory_order_release); }
    }

    std::array<std::atomic<bool>, kMaxDevices> applied_{};
    std::mutex mutex_;
};

} // namespace ninfer::ops::detail
