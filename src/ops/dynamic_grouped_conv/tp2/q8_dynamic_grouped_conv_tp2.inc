// Translation-unit fragment of src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu,
// included once at its end (after its closing namespace): the finish half of the materialized route
// as its own Op (ninfer/ops/tp2/dynamic_grouped_conv.h), launching that file's finish_kernel.
// Not a header. Our file; see docs/maintainer/upstream-merge.md.

#include "ninfer/ops/tp2/dynamic_grouped_conv.h"

#include <cstdint>
#include <string>

namespace ninfer::ops {
namespace {

void require_conv_tensor(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                         std::int32_t n3, const char* label) {
    if (t.dtype != DType::BF16 || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 ||
        t.ne[3] != n3 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("dynamic_grouped_conv_finish_add: invalid ") +
                                    label);
    }
}

} // namespace

void dynamic_grouped_conv_finish_add(const Tensor& projected, const Tensor& base_kernel,
                                     const Tensor& finish_delta, Tensor& residual,
                                     cudaStream_t stream) {
    const std::int32_t width = projected.ne[1];
    const std::int32_t batch = projected.ne[2];
    if (width < 2 || width > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument("dynamic_grouped_conv_finish_add: W must be 2..16, B 1..8");
    }
    require_conv_tensor(projected, detail::kRows, width, batch, 1, "projected");
    require_conv_tensor(residual, detail::kRows, width, batch, 1, "residual");
    require_conv_tensor(base_kernel, detail::kRows, 2, 2, 1, "base kernel");
    require_conv_tensor(finish_delta, detail::kGroups, 2, width, batch, "finish delta");
    const auto begin = [](const Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data); };
    const auto end   = [&](const Tensor& t) { return begin(t) + t.bytes(); };
    for (const Tensor* other : {&projected, &base_kernel, &finish_delta}) {
        if (begin(residual) < end(*other) && begin(*other) < end(residual)) {
            throw std::invalid_argument("dynamic_grouped_conv_finish_add: residual overlaps");
        }
    }
    const dim3 grid((detail::kRows + 255) / 256, width * batch);
    detail::finish_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data),
        static_cast<const __nv_bfloat16*>(finish_delta.data),
        static_cast<__nv_bfloat16*>(residual.data), width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
