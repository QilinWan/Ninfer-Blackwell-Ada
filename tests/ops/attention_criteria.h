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

constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

// This tree's budgets are the ones qualified on its own devices (2x RTX 5060 Ti sm_120a and the
// Ada sm_89 build): INT8-G64 and FP8 Q quantization is an accepted production approximation whose
// FP64-oracle distance is dominated by the Q path, and the small-T / head-local routes sit above
// the fork's tighter numbers. The fork's 3.15e-3 / 1.2e-2 tightenings are a 5070 Ti / 5090
// calibration; re-qualify them on this tree's hardware (S13) before narrowing here.
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

// rk4v4-e8 stores 4-bit K/V codes like NVFP4 but with a per-token scale (finer than NVFP4's
// per-16 group) and an E8 lattice K code, so it starts from the NVFP4 criterion.
constexpr ReductionCriterion kAttentionE8Criterion{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-2,
};

// A build only runs the KV routes its architecture implements: this fork is Blackwell-only, and
// rk4v4-e8 is an Ada-era lattice codec that the build drops and the capability gate refuses
// (cmake/Sm120Compat.cmake). Suites skip the formats this build cannot execute, so `ctest` stays a
// real gate instead of asking for a route that does not exist and aborting in an exception.
[[nodiscard]] constexpr bool kv_storage_supported(KvCacheStorage storage) noexcept {
    return storage != KvCacheStorage::RK4V4E8;
}

inline ReductionCriterion attention_criterion(KvCacheStorage storage) {
    if (storage == KvCacheStorage::BFloat16) return kAttentionBf16Criterion;
    if (storage == KvCacheStorage::Int8Group64) return kAttentionInt8Criterion;
    if (storage == KvCacheStorage::Fp8E4M3Row256) return kAttentionFp8Criterion;
    if (storage == KvCacheStorage::Nvfp4Group16) return kAttentionNvfp4Criterion;
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) return kAttentionK8V4Criterion;
    if (storage == KvCacheStorage::RK4V4E8) return kAttentionE8Criterion;
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
