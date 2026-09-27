#pragma once

// ninfer::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

void argmax_split_pack_launch(const Tensor& logits, const Tensor& local, int rank,
                              Tensor& candidates, cudaStream_t stream);

void argmax_split_select_launch(const Tensor& candidates, std::int32_t rows_per_rank, Tensor& out,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
