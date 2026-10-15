#pragma once

// The causal Softmax Attention Op's registered numerical criteria, shared by its conformance suite
// (softmax_attention/causal_cache.cpp) and the two-device head-local suite
// (test_attention_headlocal.cpp), so that a head-local geometry is judged by the contract of the
// storage it reads.
//
// Each is a distance to softmax_attention/oracle.h's FP64 reference over one output block. The
// conformance suite applies one criterion per registered KV-cache storage profile; token count,
// geometry, execution envelope and private launch route do not select or relax it.

#include "ninfer/types.h"
#include "ops/op_check.h"

#include <stdexcept>

namespace ninfer::test {

// Native INT8-G64 / FP8 Q quantization is an accepted production approximation. The unchanged
// FP64 oracle includes its error. Small, unit-RMS and RMS1.8 Q/K fixtures qualify that budget:
// observed relative-L2 maxima are 1.89% / 6.20%, with finite headroom below 2% / 8%.
// These are conformance-domain budgets, not bounds for arbitrary BF16 inputs or KV quality scores.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 2.0e-2,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ 4.0e-2,
};

constexpr ReductionCriterion kAttentionFp8Criterion{
    /*relative_l2*/ 8.0e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-1,
};

constexpr ReductionCriterion kAttentionNvfp4Criterion{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-2,
};

constexpr ReductionCriterion kAttentionK8V4Criterion{
    /*relative_l2*/ 8.0e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-1,
};

inline ReductionCriterion attention_criterion(KvCacheStorage storage) {
    if (storage == KvCacheStorage::BFloat16) return kAttentionBf16Criterion;
    if (storage == KvCacheStorage::Int8Group64) return kAttentionInt8Criterion;
    if (storage == KvCacheStorage::Fp8E4M3Row256) return kAttentionFp8Criterion;
    if (storage == KvCacheStorage::Nvfp4Group16) return kAttentionNvfp4Criterion;
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) return kAttentionK8V4Criterion;
    throw std::logic_error("unregistered causal-attention test storage");
}

// Two geometries that each sit within attention_criterion(storage) of the oracle, on either side,
// differ by at most twice it (triangle inequality).
inline ReductionCriterion attention_parity_criterion(KvCacheStorage storage) {
    const ReductionCriterion oracle = attention_criterion(storage);
    return {2 * oracle.relative_l2, 2 * oracle.gross_absolute,
            2 * oracle.gross_relative_to_max_reference};
}

} // namespace ninfer::test
