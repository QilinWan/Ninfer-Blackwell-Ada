#pragma once

// Helpers shared by the two-device split suites (test_*_split.cpp) and the all-reduce suite: rank
// selection and synchronization, the patterned-weight fixture with a shard origin, its upload, and
// the BF16 comparison against the single-device result.

#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "core/weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace ninfer::test {

// The BF16 unit roundoff u = 2^-8 = 1/256, not an ulp: the largest relative error of one
// round-to-nearest to BF16, whose significand has 8 bits. u * x is half an ulp of x when x is a
// power of two and approaches one ulp just below the next one, so a bound of 2u relative to a
// maximum x is one to two ulp of x.
constexpr double kBf16UnitRoundoff = 1.0 / 256.0;

inline const char* policy_name(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return "A16Only";
    case ops::LinearPolicy::AllowA8:
        return "AllowA8";
    case ops::LinearPolicy::AllowA4:
        return "AllowA4";
    }
    return "?";
}

inline void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

inline void synchronize_both(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }
}

// Uploads, poison fills and other cudaMemcpy/cudaMemset staging run on each device's legacy
// default stream. DeviceContext::stream is created with cudaStreamNonBlocking and does not order
// against it, so the staging writes are retired before a split form or a collective reads them.
// This is the caller obligation the Op contracts document; omitting the wait makes a suite
// intermittently read pre-staging bytes.
inline void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

// The block of one logical patterned weight at [row_origin, row_origin + n) x
// [column_origin, column_origin + k): a parent at origin 0, or a standalone shard holding the
// parent's block (quantized_weight.h keys every code and scale on the global coordinate).
inline quantized_weight::PackedWeight make_weight(QType qtype, std::int32_t n, std::int32_t k,
                                                  std::uint32_t seed, std::int32_t row_origin,
                                                  std::int32_t column_origin) {
    quantized_weight::PatternedWeightOptions options;
    options.row_origin              = row_origin;
    options.column_origin           = column_origin;
    options.decorrelate_coordinates = true;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    }
    return quantized_weight::make_patterned_weight(qtype, n, k, seed, options);
}

struct RankWeight {
    DeviceBuffer payload;
    Weight weight{};
};

// On the current device.
inline RankWeight upload(const quantized_weight::PackedWeight& packed) {
    RankWeight result;
    result.payload = to_device(packed.payload);
    result.weight  = packed.device_weight(result.payload.p);
    return result;
}

inline int compare(const std::string& label, const std::vector<double>& got,
                   const std::vector<double>& expected, const ReductionCriterion& criterion) {
    const ReductionStats stats =
        compute_reduction_stats(got.data(), expected.data(), static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2
              << " gross_limit=" << gross_error_limit(stats, criterion) << '\n';
    return verify_reduction(label, got, expected, criterion);
}

} // namespace ninfer::test
