#pragma once

#include "core/device.h" // CUDA_CHECK

#include <cuda_runtime.h>

#include <cstdio>

namespace ninfer {

// Saves the calling thread's current CUDA device and restores it when the scope ends, including
// during exception unwinding. Kernel launches, stream-ordered copies and cudaMalloc target the
// current device, so work for another device (tensor-parallel rank 1, a mirror) is issued inside
// one of these scopes. A failed restore cannot throw from the destructor; it is reported on stderr.
class ScopedCurrentDevice {
public:
    ScopedCurrentDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    // Saves the current device, then makes `device` current.
    explicit ScopedCurrentDevice(int device) : ScopedCurrentDevice() { select(device); }

    ~ScopedCurrentDevice() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    ScopedCurrentDevice(const ScopedCurrentDevice&)            = delete;
    ScopedCurrentDevice& operator=(const ScopedCurrentDevice&) = delete;

    // Makes `device` current until the next select() or the end of the scope.
    static void select(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

} // namespace ninfer
