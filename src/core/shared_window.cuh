#pragma once

#include "core/device.h"

#include <cuda_runtime.h>
#include <cstdint>
#include "core/kernel_attr_once.h"

// The single authority for shared staging that does not fit the static window.
//
// Static __shared__ is capped at 48 KB per CTA; both supported architectures opt in to a dynamic
// window of up to 99 KB (sm_120a and sm_89 advertise the same per-CTA limit), so a schedule whose
// staging union exceeds the static cap addresses identical storage through the dynamic window
// instead. The tile shape, and therefore the split-K reduction order and the numbers, is unchanged:
// only the storage class moves. A kernel at or below the cap keeps its static allocation, so no
// attribute request is made and the launch passes zero dynamic bytes.
//
// Core owns this because it is a physical device primitive shared by every Op family: the Q8 MMA,
// sliced-K and grouped families, the projection families that instantiate them, and
// context_kv_materialize all follow the same rule.
namespace ninfer::core {

// The cap a schedule must go dynamic above, and the largest window an opt-in may request.
inline constexpr int kSharedStaticCapBytes = 48 * 1024;
inline constexpr int kSharedWindowMaxBytes = 99 * 1024;

// Storage for a staging union of `Bytes`, as a 16-byte aligned block of the window.
template <int Bytes>
__device__ __forceinline__ unsigned char* shared_window_storage() {
    static_assert(Bytes <= kSharedWindowMaxBytes,
                  "shared staging exceeds the per-CTA dynamic window both architectures opt in to");
    if constexpr (Bytes > kSharedStaticCapBytes) {
        extern __shared__ __align__(16) unsigned char dynamic_shared[];
        return dynamic_shared;
    } else {
        __shared__ __align__(16) unsigned char storage[Bytes];
        return storage;
    }
}

// Opt one instantiated kernel in to a dynamic window of `Bytes` and return the size its launch must
// pass, so a launcher writes one expression for both regimes. Each expansion owns its own once-flag,
// so the attribute call runs once per instantiated kernel rather than once per launch.
template <int Bytes, auto Kernel>
int request_shared_window() {
    if constexpr (Bytes > kSharedStaticCapBytes) {
        static core::FuncAttrPerDevice shared_window_status;
        shared_window_status.ensure(
            Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        return Bytes;
    } else {
        return 0;
    }
}

} // namespace ninfer::core

// The kernel expression is last because its own template argument list contains commas.
#define NINFER_REQUEST_SHARED_WINDOW(bytes, ...)                                                   \
    do {                                                                                           \
        if constexpr ((bytes) != 0) {                                                              \
            static const ::cudaError_t shared_window_status =                                      \
                ::cudaFuncSetAttribute(__VA_ARGS__, ::cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                       static_cast<int>(bytes));                                   \
            CUDA_CHECK(shared_window_status);                                                      \
        }                                                                                          \
    } while (0)
