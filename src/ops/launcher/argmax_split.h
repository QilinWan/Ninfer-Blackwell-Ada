#pragma once

// ninfer::ops::detail - private launch prototypes of the two-device split argmax
// (ninfer/ops/tp2/argmax.h). Our file: ops/launcher/argmax.h is upstream's.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void argmax_split_pack_launch(const Tensor& logits, const Tensor& local, int rank,
                              Tensor& candidates, cudaStream_t stream);

void argmax_split_select_launch(const Tensor& candidates, std::int32_t rows_per_rank, Tensor& out,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
