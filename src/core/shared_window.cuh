#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

// Shared staging larger than the 48 KB static window has to come from the dynamic window, which
// needs a per-kernel opt-in before launch; the Op families already gate on that limit (see
// linear/bf16/bf16_launch.cuh). A zero byte count means the kernel keeps its static allocation, so
// no request is made. Each expansion owns its once-flag, so the attribute call runs once per
// instantiated kernel rather than once per launch. The kernel expression is last because its
// template argument list contains commas.
#define NINFER_REQUEST_SHARED_WINDOW(bytes, ...)                                                   \
    do {                                                                                           \
        if constexpr ((bytes) != 0) {                                                              \
            static const ::cudaError_t shared_window_status =                                      \
                ::cudaFuncSetAttribute(__VA_ARGS__, ::cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                       static_cast<int>(bytes));                                   \
            CUDA_CHECK(shared_window_status);                                                      \
        }                                                                                          \
    } while (0)
