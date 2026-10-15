#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The validation and per-format dispatch behind ops::linear(), for sibling Op families that issue
// a plain projection with a nullable workspace. The public overloads are thin wrappers over this
// pair.
void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy);

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
