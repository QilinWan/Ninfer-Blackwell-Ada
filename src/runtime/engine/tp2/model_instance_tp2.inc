// Translation-unit fragment of src/runtime/engine/model_instance.cpp, included once at its end
// (after its closing namespace): the Engine option checks of the two-device mode. Not a header.
// Our file; see docs/maintainer/upstream-merge.md.

namespace ninfer::runtime {
namespace {

void validate_tensor_parallel_options(const EngineOptions& options) {
    // Rejected before the artifact is read: the two-device schedule covers text and
    // multimodal prefill, the ordinary decode round, the MTP round and the DFlash2 round of
    // the dense Text model, and causal scoring over its text prefill, only
    // (models/qwen3_5/execution/text.h).
    if (options.speculative.backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("Engine tp 2 does not support DFlash speculative decoding");
    }
    // The DFlash2 drafter runs whole on rank 0 and ranks its candidates over one complete
    // proposal head; the full output head is split by vocabulary across the ranks, so only
    // the optimized head, which rank 0 holds whole, can serve it.
    if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
        options.speculative.proposal_head != ProposalHead::Optimized) {
        throw std::invalid_argument(
            "Engine tp 2 DFlash2 requires the optimized proposal head (--lm-head-draft)");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16 &&
        options.kv_cache != KvCacheStorage::Int8Group64 &&
        options.kv_cache != KvCacheStorage::Nvfp4Group16) {
        throw std::invalid_argument("Engine tp 2 supports only bf16, int8 and nvfp4 KV caches");
    }
    if (options.context_cache.host_state_slots != 0 ||
        options.context_cache.host_kv_capacity_bytes != 0) {
        throw std::invalid_argument(
            "Engine tp 2 requires context_cache host_state_slots and host_kv_capacity_bytes "
            "of 0: rank 1's KV and state have no Host tier");
    }
}

} // namespace
} // namespace ninfer::runtime
