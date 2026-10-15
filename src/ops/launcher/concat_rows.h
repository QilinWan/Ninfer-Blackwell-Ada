#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Concatenates two contiguous BF16 [rows, columns] blocks along ne[0] into a contiguous
// [rows0 + rows1, columns] destination: for every column c, destination[0, rows0) of c is first's
// column c and destination[rows0, rows0 + rows1) is second's. All three live on the stream's device
// and must not overlap. A kernel rather than two pitched copies because a captured 2D memcpy node
// cannot be updated in place once its extent or pointers change (cudaGraphExecUpdate rejects it
// with ParametersChanged), while a kernel node can.
void concat_rows_bf16_launch(const Tensor& first, const Tensor& second, Tensor& destination,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
