#pragma once

// E8 (rk4v4-e8) output un-rotation.
//
// The E8 route rotates Q, K and V with the normalized D256 Sylvester transform (x2^-4), so the
// QK scores are rotation invariant but the PV result lands in the rotated domain:
//
//     V_stored = H * V ,  O_rot = sum_j p_j * (H * v_j) = H * O_true
//
// H is orthonormal and symmetric, so H^-1 == H and re-applying the same transform recovers
// O_true.  The reference Ada port does exactly this with kv_cache_inverse_rotate_output_kernel
// (src/ops/kv_cache/int8_g64_codec.cuh there, called from prompt.cu and small_t.cu); the sm_89
// port that this file belongs to rotates a whole 256-dim row instead of four 64-dim groups, so
// the un-rotation has to cover the full row.
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {
namespace detail {

// One warp per (q_head, row) unit; lane l carries dimensions l + 32*r, the layout
// normalized_hadamard_d256_inplace expects.  `out` is [D256][QHeads][width][batch].
template <int QHeads, int HeadDim>
__global__ void e8_inverse_rotate_output_kernel(__nv_bfloat16* __restrict__ output, int width,
                                                int full_width, int column_begin,
                                                const std::int32_t* __restrict__ valid_columns) {
    const int lane = static_cast<int>(threadIdx.x);
    if (lane >= 32) { return; }
    const int unit   = static_cast<int>(blockIdx.x);
    const int q_head = unit % QHeads;
    const int row    = unit / QHeads;
    const int batch  = row / width;
    const int token  = row - batch * width;
    const int column = column_begin + token;
    if (token >= width || (valid_columns != nullptr && column >= valid_columns[batch])) { return; }

    const std::int64_t base =
        static_cast<std::int64_t>(HeadDim) *
        (static_cast<std::int64_t>(q_head) +
         static_cast<std::int64_t>(QHeads) *
             (static_cast<std::int64_t>(column) + static_cast<std::int64_t>(full_width) * batch));

    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        values[r] = __bfloat162float(output[base + lane + 32 * r]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        output[base + lane + 32 * r] = __float2bfloat16(values[r]);
    }
}

template <int QHeads, int HeadDim>
void e8_inverse_rotate_output_launch(Tensor& out, std::int32_t width, std::int32_t full_width,
                                     std::int32_t column_begin, const Tensor* valid_columns,
                                     std::int32_t batch_size, cudaStream_t stream) {
    if (width <= 0 || batch_size <= 0) { return; }
    const int units = QHeads * width * batch_size;
    e8_inverse_rotate_output_kernel<QHeads, HeadDim><<<units, 32, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), width, full_width, column_begin,
        valid_columns == nullptr ? nullptr
                                 : static_cast<const std::int32_t*>(valid_columns->data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace detail
} // namespace ninfer::ops
