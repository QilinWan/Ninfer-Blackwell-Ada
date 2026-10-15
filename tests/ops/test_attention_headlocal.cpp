// Two-device head-local parity of causal Softmax Attention.
//
// Grouped-query attention is separable in the head index: query head h reads only KV head h/6 and
// takes its Softmax over that head's keys. Under two-device tensor parallelism device r therefore
// owns query heads [12r,12r+12) and KV heads [2r,2r+2) of the 24/4 model, keeps a KV cache holding
// only its two heads, and runs the 12/2 geometry with no communication. The group size stays 6.
//
// Each case runs the same logical sequence twice: once through the 24/4 geometry on device 0, and
// once through the 12/2 geometry on each device with that device's query and KV heads. A sequence
// prefills its history through causal_softmax_attention(), which appends the history rows, then
// attends a block of W new tokens for every table row in the batch; single-row cases then also
// read the populated cache back through causal_softmax_attention_cached(). Every 12/2 output is
// compared with the 24/4 output of the same global heads. A wrong head mapping still produces
// well-formed output, so only this per-head comparison can catch it.
//
// Parity is not bit-exact. Both geometries read identical cache rows (the K/V codecs work per
// token and KV head), but 12/2 plans its own KV partition for two KV heads instead of four, so the
// keys are split and the partial Softmax states summed in a different order, and the two FP32
// results can round to adjacent BF16 outputs. One BF16 ulp can exceed the storage's oracle
// criterion C itself: at an output of 0.26 the ulp is 1.95e-3, while the BF16 gross limit
// 1.0e-3 + 2.7e-3 * max|reference| stays below it for any maximum under 0.35. The causal Softmax
// Attention suite qualifies each geometry against the FP64 oracle with C, so by the triangle
// inequality two qualified outputs differ by at most 2C, which is the parity bound here. A wrong
// head mapping or cache row gives errors of the outputs' own magnitude, far above 2C.
//
// The tiled prefill kernels and the INT8 grouped kernels opt into more than 48 KiB of dynamic
// shared memory, a per-device attribute, so the prefills on device 1 also check that the opt-in
// reaches the second device.
//
// Every case needs two CUDA devices in one process and the suite reports 77 with fewer.
#include "ninfer/ops/softmax_attention.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "ops/attention_criteria.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim       = 256;
constexpr float kAttentionScale       = 0.0625F;
constexpr std::int32_t kGlobalQHeads  = 24;
constexpr std::int32_t kGlobalKvHeads = 4;
constexpr std::int32_t kLocalQHeads   = kGlobalQHeads / 2;
constexpr std::int32_t kLocalKvHeads  = kGlobalKvHeads / 2;

struct Case {
    KvCacheStorage storage;
    std::int32_t batch;   // table rows attended together
    std::int32_t history; // tokens already appended to every row
    std::int32_t width;   // new tokens per row
};

const char* storage_name(KvCacheStorage storage) {
    return storage == KvCacheStorage::BFloat16 ? "bf16" : storage == KvCacheStorage::Int8Group64 ? "int8" : "nvfp4";
}

// Paged K/V planes and one block-table row per sequence, on the current device. Logical page p of
// row b lives on physical page 2*(b*pages+p)+1, so the table indirection is never the identity.
class PagedCache {
public:
    PagedCache(KvCacheStorage storage, std::int32_t kv_heads, std::int32_t rows,
               std::int32_t tokens)
        : storage_(storage), kv_heads_(kv_heads), rows_(rows),
          logical_pages_((tokens + kPagedKVPageSize - 1) / kPagedKVPageSize),
          physical_pages_(2 * rows * logical_pages_ + 1),
          layout_(paged_kv_storage_layout(storage, kHeadDim)) {
        k_       = plane(layout_.key.data_dtype, layout_.key.data_leading_extent);
        v_       = plane(layout_.value.data_dtype, layout_.value.data_leading_extent);
        k_scale_ = plane(layout_.key.scale_dtype, layout_.key.scale_leading_extent);
        v_scale_ = plane(layout_.value.scale_dtype, layout_.value.scale_leading_extent);
        std::vector<std::int32_t> tables(static_cast<std::size_t>(rows) * logical_pages_);
        for (std::size_t page = 0; page < tables.size(); ++page) {
            tables[page] = 2 * static_cast<std::int32_t>(page) + 1;
        }
        tables_ = to_device(tables);
    }

    [[nodiscard]] PagedKVBatchLayerView batch_view() const {
        return {
            plane_tensor(k_, layout_.key.data_dtype, layout_.key.data_leading_extent),
            plane_tensor(v_, layout_.value.data_dtype, layout_.value.data_leading_extent),
            plane_tensor(k_scale_, layout_.key.scale_dtype, layout_.key.scale_leading_extent),
            plane_tensor(v_scale_, layout_.value.scale_dtype, layout_.value.scale_leading_extent),
            Tensor(tables_.p, DType::I32, {logical_pages_, rows_}),
            kHeadDim,
            kv_heads_,
            storage_};
    }

    [[nodiscard]] PagedKVLayerView row_view(std::int32_t row) const {
        const PagedKVBatchLayerView batch = batch_view();
        auto* table                       = static_cast<std::int32_t*>(tables_.p) +
                                            static_cast<std::ptrdiff_t>(row) * logical_pages_;
        return {batch.k_pages,
                batch.v_pages,
                batch.k_scale_pages,
                batch.v_scale_pages,
                Tensor(table, DType::I32, {logical_pages_}),
                kHeadDim,
                kv_heads_,
                storage_};
    }

private:
    [[nodiscard]] std::optional<DeviceBuffer> plane(DType dtype, std::int32_t leading) const {
        if (leading == 0) { return std::nullopt; }
        std::optional<DeviceBuffer> buffer;
        buffer.emplace(static_cast<std::size_t>(leading) * kPagedKVPageSize * kv_heads_ *
                       physical_pages_ * dtype_size(dtype));
        buffer->fill(0);
        return buffer;
    }

    [[nodiscard]] Tensor plane_tensor(const std::optional<DeviceBuffer>& buffer, DType dtype,
                                      std::int32_t leading) const {
        if (!buffer) { return Tensor{}; }
        return Tensor(buffer->p, dtype, {leading, kPagedKVPageSize, kv_heads_, physical_pages_});
    }

    KvCacheStorage storage_;
    std::int32_t kv_heads_;
    std::int32_t rows_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    PagedKVStorageLayout layout_;
    std::optional<DeviceBuffer> k_;
    std::optional<DeviceBuffer> v_;
    std::optional<DeviceBuffer> k_scale_;
    std::optional<DeviceBuffer> v_scale_;
    DeviceBuffer tables_;
};

// Head-major BF16 activations [256, heads, tokens] of one logical sequence.
struct Sequence {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::int32_t tokens = 0;
};

std::vector<float> make_values(std::size_t count, std::uint32_t seed, float bound) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -bound, bound);
    round_to_bf16(values);
    return values;
}

Sequence make_sequence(std::int32_t tokens, std::uint32_t seed) {
    const auto q_count  = static_cast<std::size_t>(kHeadDim) * kGlobalQHeads * tokens;
    const auto kv_count = static_cast<std::size_t>(kHeadDim) * kGlobalKvHeads * tokens;
    return {make_values(q_count, seed, 0.25F), make_values(kv_count, seed + 1U, 0.25F),
            make_values(kv_count, seed + 2U, 1.0F), tokens};
}

// Heads [begin, begin + count) of every token of a [256, heads, tokens] tensor.
template <typename T>
std::vector<T> slice_heads(const std::vector<T>& source, std::int32_t heads, std::int32_t begin,
                           std::int32_t count, std::int32_t tokens) {
    const auto head_elements = static_cast<std::ptrdiff_t>(kHeadDim);
    std::vector<T> result(static_cast<std::size_t>(kHeadDim) * count * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const auto from = source.begin() + head_elements * (std::ptrdiff_t{token} * heads + begin);
        std::copy(from, from + head_elements * count,
                  result.begin() + head_elements * count * token);
    }
    return result;
}

// Concatenates per-row [256, heads, tokens] tensors into [256, heads, tokens, rows].
std::vector<float> concat_rows(const std::vector<std::vector<float>>& rows) {
    std::vector<float> result;
    for (const std::vector<float>& row : rows) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

// Outputs of one geometry's run of a case, as FP64 [256, heads, tokens(, rows)].
struct RunOutputs {
    std::vector<std::vector<double>> history; // per row, [256, heads, history]
    std::vector<double> block;                // [256, heads, width, batch]
    std::vector<double> cached;               // [256, heads, width]; single-row cases only
};

class Runner {
public:
    int guard_failures = 0;

    Runner(const DeviceContext& device, ops::AttentionHeadGeometry geometry, KvCacheStorage storage)
        : device_(device), geometry_(geometry), storage_(storage) {}

    // Runs `test_case` for heads [q_begin, q_begin + Hq) and KV heads [kv_begin, kv_begin + Hkv) of
    // `history`/`block` rows. Leaves the device current.
    RunOutputs run(const Case& test_case, const std::vector<Sequence>& history,
                   const std::vector<Sequence>& block, std::int32_t q_begin,
                   std::int32_t kv_begin) {
        cuda_check(cudaSetDevice(device_.device), "cudaSetDevice");
        const std::int32_t hq    = geometry_.query_heads;
        const std::int32_t hkv   = geometry_.kv_heads;
        const std::int32_t total = test_case.history + test_case.width;
        const PagedCache cache(storage_, hkv, test_case.batch, total);
        RunOutputs outputs;

        for (std::int32_t row = 0; row < test_case.batch && test_case.history > 0; ++row) {
            const Sequence& sequence = history[static_cast<std::size_t>(row)];
            std::vector<std::int32_t> positions(static_cast<std::size_t>(sequence.tokens));
            for (std::int32_t token = 0; token < sequence.tokens; ++token) {
                positions[static_cast<std::size_t>(token)] = token;
            }
            outputs.history.push_back(
                attend(slice_heads(sequence.q, kGlobalQHeads, q_begin, hq, sequence.tokens),
                       slice_heads(sequence.k, kGlobalKvHeads, kv_begin, hkv, sequence.tokens),
                       slice_heads(sequence.v, kGlobalKvHeads, kv_begin, hkv, sequence.tokens),
                       positions, {row}, sequence.tokens, 1, cache));
        }

        std::vector<std::vector<float>> q_rows;
        std::vector<std::vector<float>> k_rows;
        std::vector<std::vector<float>> v_rows;
        std::vector<std::int32_t> positions;
        std::vector<std::int32_t> table_rows;
        for (std::int32_t row = 0; row < test_case.batch; ++row) {
            const Sequence& sequence = block[static_cast<std::size_t>(row)];
            q_rows.push_back(slice_heads(sequence.q, kGlobalQHeads, q_begin, hq, test_case.width));
            k_rows.push_back(
                slice_heads(sequence.k, kGlobalKvHeads, kv_begin, hkv, test_case.width));
            v_rows.push_back(
                slice_heads(sequence.v, kGlobalKvHeads, kv_begin, hkv, test_case.width));
            for (std::int32_t token = 0; token < test_case.width; ++token) {
                positions.push_back(test_case.history + token);
            }
            table_rows.push_back(row);
        }
        outputs.block = attend(concat_rows(q_rows), concat_rows(k_rows), concat_rows(v_rows),
                               positions, table_rows, test_case.width, test_case.batch, cache);
        if (test_case.batch == 1) {
            outputs.cached = attend_cached(q_rows.front(), positions, test_case.width, cache);
        }
        return outputs;
    }

private:
    [[nodiscard]] std::size_t workspace_bytes(ops::CausalAttentionExecutionEnvelope envelope,
                                              std::int32_t width, std::int32_t batch) const {
        return std::max<std::size_t>(ops::causal_softmax_attention_workspace_capacity_bytes(
                                         geometry_, storage_, envelope, batch, width, width),
                                     256);
    }

    std::vector<double> attend(const std::vector<float>& q, const std::vector<float>& k,
                               const std::vector<float>& v,
                               const std::vector<std::int32_t>& positions,
                               const std::vector<std::int32_t>& table_rows, std::int32_t width,
                               std::int32_t batch, const PagedCache& cache) {
        const std::int32_t hq  = geometry_.query_heads;
        const std::int32_t hkv = geometry_.kv_heads;
        const DeviceBuffer dq  = to_device_bf16(q);
        const DeviceBuffer dk  = to_device_bf16(k);
        const DeviceBuffer dv  = to_device_bf16(v);
        const DeviceBuffer dp  = to_device(positions);
        const DeviceBuffer dt  = to_device(table_rows);
        GuardedDeviceBuffer out(q.size() * sizeof(std::uint16_t));
        out.fill(0xff);
        const std::int32_t visible = *std::max_element(positions.begin(), positions.end()) + 1;
        const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(visible),
                                                             static_cast<std::uint32_t>(visible)};
        DeviceArena workspace(workspace_bytes(envelope, width, batch));
        const Tensor tq(dq.p, DType::BF16, {kHeadDim, hq, width, batch});
        const Tensor tk(dk.p, DType::BF16, {kHeadDim, hkv, width, batch});
        const Tensor tv(dv.p, DType::BF16, {kHeadDim, hkv, width, batch});
        const Tensor tp(dp.p, DType::I32, {width, batch});
        const Tensor tt(dt.p, DType::I32, {batch});
        Tensor tout(out.data(), DType::BF16, {kHeadDim, hq, width, batch});
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, tt, geometry_, kAttentionScale,
                                      cache.batch_view(), envelope, workspace, tout,
                                      device_.stream);
        cuda_check(cudaStreamSynchronize(device_.stream), "cudaStreamSynchronize");
        guard_failures += out.verify_guards("causal_softmax_attention output");
        return from_device_bf16(out.data(), q.size());
    }

    std::vector<double> attend_cached(const std::vector<float>& q,
                                      const std::vector<std::int32_t>& positions,
                                      std::int32_t width, const PagedCache& cache) {
        const std::int32_t hq = geometry_.query_heads;
        const DeviceBuffer dq = to_device_bf16(q);
        const DeviceBuffer dp = to_device(positions);
        GuardedDeviceBuffer out(q.size() * sizeof(std::uint16_t));
        out.fill(0xff);
        const auto visible = static_cast<std::uint32_t>(positions.back() + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        DeviceArena workspace(workspace_bytes(envelope, width, 1));
        const Tensor tq(dq.p, DType::BF16, {kHeadDim, hq, width});
        const Tensor tp(dp.p, DType::I32, {width});
        Tensor tout(out.data(), DType::BF16, {kHeadDim, hq, width});
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::causal_softmax_attention_cached(tq, tp, geometry_, kAttentionScale, cache.row_view(0),
                                             envelope, workspace, tout, device_.stream);
        cuda_check(cudaStreamSynchronize(device_.stream), "cudaStreamSynchronize");
        guard_failures += out.verify_guards("causal_softmax_attention_cached output");
        return from_device_bf16(out.data(), q.size());
    }

    const DeviceContext& device_;
    ops::AttentionHeadGeometry geometry_;
    KvCacheStorage storage_;
};

int compare(const std::string& label, const std::vector<double>& local,
            const std::vector<double>& global, std::int32_t rank, std::int32_t tokens,
            const ReductionCriterion& criterion) {
    const std::vector<double> expected =
        slice_heads(global, kGlobalQHeads, rank * kLocalQHeads, kLocalQHeads, tokens);
    return verify_reduction(label, local, expected, criterion);
}

int run_case(const ExecutionContext& ec, const Case& test_case, std::uint32_t seed) {
    const std::string head =
        std::string(storage_name(test_case.storage)) + " B=" + std::to_string(test_case.batch) +
        " history=" + std::to_string(test_case.history) + " W=" + std::to_string(test_case.width);
    const ReductionCriterion criterion = attention_parity_criterion(test_case.storage);

    std::vector<Sequence> history;
    std::vector<Sequence> block;
    for (std::int32_t row = 0; row < test_case.batch; ++row) {
        const auto row_seed = seed + 16U * static_cast<std::uint32_t>(row);
        history.push_back(make_sequence(test_case.history, row_seed));
        block.push_back(make_sequence(test_case.width, row_seed + 8U));
    }

    Runner global(*ec.dev[0], {kHeadDim, kGlobalQHeads, kGlobalKvHeads}, test_case.storage);
    const RunOutputs reference = global.run(test_case, history, block, 0, 0);
    int failures               = global.guard_failures;

    for (std::int32_t rank = 0; rank < 2; ++rank) {
        Runner local(*ec.dev[rank], {kHeadDim, kLocalQHeads, kLocalKvHeads}, test_case.storage);
        const RunOutputs observed =
            local.run(test_case, history, block, rank * kLocalQHeads, rank * kLocalKvHeads);
        failures += local.guard_failures;
        const std::string label = head + " rank " + std::to_string(rank);
        for (std::size_t row = 0; row < observed.history.size(); ++row) {
            failures +=
                compare(label + " history row " + std::to_string(row), observed.history[row],
                        reference.history[row], rank, test_case.history, criterion);
        }
        failures += compare(label + " block", observed.block, reference.block, rank,
                            test_case.width * test_case.batch, criterion);
        if (!observed.cached.empty()) {
            failures += compare(label + " cached", observed.cached, reference.cached, rank,
                                test_case.width, criterion);
        }
    }
    std::cout << (failures ? "FAIL " : "OK ") << head << '\n';
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: head-local attention parity requires two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    int failures = 0;
    try {
        const ExecutionContext ec({0, 1});
        std::uint32_t seed = 900U;
        for (const KvCacheStorage storage :
             {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16}) {
            // Decode, grouped and parallel-grouped blocks on both sides of the 8-token seam,
            // tiled prompt blocks, long-context decode, and decode/verify batches, each after a
            // history appended through the Op itself.
            const std::vector<Case> cases{
                {storage, 1, 64, 1},   {storage, 1, 61, 6},   {storage, 1, 63, 7},
                {storage, 1, 61, 8},   {storage, 1, 61, 9},   {storage, 1, 127, 16},
                {storage, 1, 0, 67},   {storage, 1, 129, 65}, {storage, 1, 17, 257},
                {storage, 1, 2048, 1}, {storage, 4, 127, 1},  {storage, 2, 61, 16},
            };
            for (const Case& test_case : cases) {
                failures += run_case(ec, test_case, seed);
                seed += 101U;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "head-local attention: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " head-local attention\n";
    return failures ? 1 : 0;
}
