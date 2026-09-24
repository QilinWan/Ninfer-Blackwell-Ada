#include "ops/launcher/concat_rows.h"

#include "core/device.h"
#include "ops/common/math.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kGridCap = 4096;

// `rows0`, `rows1` and the indices count elements of T: 16-byte vectors of eight BF16 values when
// both blocks' row counts allow it, single BF16 values otherwise.
template <class T>
__global__ void concat_rows_kernel(const T* __restrict__ first, const T* __restrict__ second,
                                   T* __restrict__ destination, std::int64_t rows0,
                                   std::int64_t rows1, std::int64_t columns) {
    const std::int64_t rows  = rows0 + rows1;
    const std::int64_t total = rows * columns;
    const std::int64_t step  = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += step) {
        const std::int64_t column = i / rows;
        const std::int64_t row    = i - column * rows;
        destination[i] = row < rows0 ? first[column * rows0 + row]
                                     : second[column * rows1 + (row - rows0)];
    }
}

template <class T>
void launch(const Tensor& first, const Tensor& second, Tensor& destination, std::int64_t rows0,
            std::int64_t rows1, std::int64_t columns, cudaStream_t stream) {
    const std::int64_t total = (rows0 + rows1) * columns;
    const int grid           = static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(div_up(total, static_cast<std::int64_t>(kBlock)), kGridCap)));
    concat_rows_kernel<T><<<grid, kBlock, 0, stream>>>(static_cast<const T*>(first.data),
                                                         static_cast<const T*>(second.data),
                                                         static_cast<T*>(destination.data), rows0,
                                                         rows1, columns);
}

} // namespace

void concat_rows_bf16_launch(const Tensor& first, const Tensor& second, Tensor& destination,
                             cudaStream_t stream) {
    const std::int64_t rows0   = first.ne[0];
    const std::int64_t rows1   = second.ne[0];
    const std::int64_t columns = first.ne[1];
    if (first.dtype != DType::BF16 || second.dtype != DType::BF16 ||
        destination.dtype != DType::BF16 || !first.is_contiguous() || !second.is_contiguous() ||
        !destination.is_contiguous() || second.ne[1] != columns ||
        destination.ne[0] != rows0 + rows1 || destination.ne[1] != columns ||
        first.ne[2] * first.ne[3] != 1 || second.ne[2] * second.ne[3] != 1 ||
        destination.ne[2] * destination.ne[3] != 1) {
        throw std::invalid_argument("concat_rows: blocks and destination do not match");
    }
    if (columns == 0 || rows0 + rows1 == 0) { return; }
    constexpr std::int64_t kVector = 8; // BF16 values per 16 bytes
    const bool aligned = (reinterpret_cast<std::uintptr_t>(first.data) & 0xfu) == 0 &&
                         (reinterpret_cast<std::uintptr_t>(second.data) & 0xfu) == 0 &&
                         (reinterpret_cast<std::uintptr_t>(destination.data) & 0xfu) == 0;
    if (aligned && rows0 % kVector == 0 && rows1 % kVector == 0) {
        launch<uint4>(first, second, destination, rows0 / kVector, rows1 / kVector, columns,
                      stream);
    } else {
        launch<std::uint16_t>(first, second, destination, rows0, rows1, columns, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
