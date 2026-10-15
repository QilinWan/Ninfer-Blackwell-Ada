// Two-device vocabulary-split top sixteen of the optimized proposal head (DFlash2 candidates).
//
// The Q4 proposal head `[131072,5120]` splits its rows into two `[65536,5120]` blocks, and its row
// -> global token map into the matching `[65536]` blocks. Each rank ranks its block
// (ops::linear_topk_split) and packs its sixteen keys (ops::topk_split_pack); one allreduce_sum of
// the packed keys gives rank 0 both lists and ops::topk_split_merge writes the complete top
// sixteen. The suite checks
//
//   1. the split result equals ops::linear_topk over the whole head, ids and FP32 score bits, at
//      the decode column counts of K=7 at C=1..4 and at the other routes' boundaries, eagerly
//      (staged copies) and replayed from one two-device graph through the mailbox;
//   2. the merge on planted keys: equal scores across and inside the ranks (the lower global id
//      wins), -0 against +0, the maximum in either rank, and the INT_MAX sentinel.
//
// The patterned Q4 fixture repeats row classes, so equal scores across the blocks occur and case 1
// exercises the id tie-break through the exchange. Every case needs two CUDA devices; the suite
// reports 77 with fewer. The registry probe is host-only and runs first.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ninfer/ops/tp2/linear_topk.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/split_test_support.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr std::int32_t kRows      = 131072;
constexpr std::int32_t kHalf      = ops::kLinearTopKSplitRows;
constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kTopK      = 16;
constexpr std::int32_t kValidRows = 248077;
constexpr QType kQType            = QType::Q4_G64_FP16;

std::uint32_t ordered_bits(float value) {
    if (value == 0.0F) { value = 0.0F; }
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000U) != 0 ? ~bits : (bits ^ 0x80000000U);
}

std::uint64_t host_key(float score, std::int32_t id) {
    if (id == INT_MAX) { return 0; }
    return (static_cast<std::uint64_t>(ordered_bits(score)) << 32) |
           static_cast<std::uint32_t>(0xffffffffU - static_cast<std::uint32_t>(id));
}

// Per-rank operands of the exchange over `columns` columns, plus rank 0's merged output.
struct SplitBuffers {
    std::array<std::optional<GuardedDeviceBuffer>, 2> ids, scores, candidates, staging;
    std::optional<GuardedDeviceBuffer> merged_ids, merged_scores;
    std::array<Tensor, 2> ids_t, scores_t, candidates_t, staging_t;
    Tensor merged_ids_t, merged_scores_t;

    SplitBuffers(const ExecutionContext& ec, std::int32_t columns) {
        const auto count  = static_cast<std::size_t>(columns);
        const auto packed = count * ops::kTopKSplitCandidateRows * sizeof(std::uint16_t);
        for (int rank = 0; rank < 2; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            ids[r].emplace(count * kTopK * sizeof(std::int32_t));
            scores[r].emplace(count * kTopK * sizeof(float));
            candidates[r].emplace(packed);
            staging[r].emplace(packed);
            candidates[r]->fill(0x7f);
            ids_t[r]        = Tensor(ids[r]->data(), DType::I32, {kTopK, columns});
            scores_t[r]     = Tensor(scores[r]->data(), DType::FP32, {kTopK, columns});
            candidates_t[r] = Tensor(candidates[r]->data(), DType::BF16,
                                     {ops::kTopKSplitCandidateRows, columns});
            staging_t[r]    = Tensor(staging[r]->data(), DType::BF16,
                                     {ops::kTopKSplitCandidateRows, columns});
        }
        set_device(ec, 0);
        merged_ids.emplace(count * kTopK * sizeof(std::int32_t));
        merged_scores.emplace(count * kTopK * sizeof(float));
        merged_ids->fill(0xcd);
        merged_scores->fill(0xcd);
        merged_ids_t    = Tensor(merged_ids->data(), DType::I32, {kTopK, columns});
        merged_scores_t = Tensor(merged_scores->data(), DType::FP32, {kTopK, columns});
    }

    int verify_guards(const ExecutionContext& ec, const std::string& label) {
        int failures = 0;
        for (int rank = 0; rank < 2; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            failures += ids[r]->verify_guards(label + " ids");
            failures += scores[r]->verify_guards(label + " scores");
            failures += candidates[r]->verify_guards(label + " candidates");
            failures += staging[r]->verify_guards(label + " staging");
        }
        set_device(ec, 0);
        return failures + merged_ids->verify_guards(label + " merged ids") +
               merged_scores->verify_guards(label + " merged scores");
    }

    struct Result {
        std::vector<std::int32_t> ids;
        std::vector<std::uint32_t> score_bits;
    };

    Result merged(const ExecutionContext& ec, std::int32_t columns) {
        set_device(ec, 0);
        const auto count = static_cast<std::size_t>(columns) * kTopK;
        return {from_device<std::int32_t>(merged_ids->data(), count),
                from_device<std::uint32_t>(merged_scores->data(), count)};
    }
};

// Everything after the ranks' local top sixteen, as execution::dflash2_candidates_split issues it.
void issue_exchange(const ExecutionContext& ec, const ops::PeerEvents& events, SplitBuffers& b) {
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        ops::topk_split_pack(b.ids_t[r], b.scores_t[r], rank, b.candidates_t[r],
                             ec.dev[r]->stream);
    }
    ops::allreduce_sum(b.candidates_t, b.staging_t, ec, events);
    set_device(ec, 0);
    ops::topk_split_merge(b.candidates_t[0], b.merged_ids_t, b.merged_scores_t,
                          ec.dev[0]->stream);
}

int verify_result(const std::string& label, const SplitBuffers::Result& got,
                  const SplitBuffers::Result& expected) {
    int failures = verify_exact((label + " ids").c_str(), got.ids, expected.ids);
    failures += verify_exact((label + " score bits").c_str(), got.score_bits, expected.score_bits);
    return failures;
}

// The exchange eagerly (staged) and replayed three times from one two-device graph through the
// mailbox, from local candidates the caller already wrote into `b` (`local` re-issues them).
template <class Local>
int check_exchange(const std::string& label, const ExecutionContext& ec, ops::PeerMailbox& mailbox,
                   std::int32_t columns, const SplitBuffers::Result& expected, Local&& local) {
    int failures = 0;
    {
        const ops::PeerEvents events(ec);
        SplitBuffers b(ec, columns);
        retire_staging(ec);
        local(b, nullptr);
        issue_exchange(ec, events, b);
        synchronize_both(ec);
        failures += verify_result(label + " eager", b.merged(ec, columns), expected);
        failures += b.verify_guards(ec, label + " eager");
    }
    ops::PeerEvents events(ec);
    events.attach_mailbox(&mailbox);
    SplitBuffers b(ec, columns);
    retire_staging(ec);
    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    set_device(ec, 0);
    definition.capture(
        ec.dev[0]->stream,
        [&] {
            local(b, &events);
            issue_exchange(ec, events, b);
            set_device(ec, 0);
        },
        DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    for (int replay = 0; replay < 3; ++replay) {
        set_device(ec, 0);
        b.merged_ids->fill(0xcd);
        b.merged_scores->fill(0xcd);
        retire_staging(ec);
        executable.launch(ec.dev[0]->stream);
        synchronize_both(ec);
        failures += verify_result(label + " mailbox replay " + std::to_string(replay),
                                  b.merged(ec, columns), expected);
    }
    if (mailbox.hang_reported()) {
        std::cerr << label << ": the mailbox reported a hang\n";
        ++failures;
    }
    return failures + b.verify_guards(ec, label + " mailbox");
}

int run_head(const ExecutionContext& ec, ops::PeerMailbox& mailbox) {
    constexpr std::uint32_t kSeed = 811U;
    int failures                  = 0;
    const qw::PackedWeight parent = make_weight(kQType, kRows, kHidden, kSeed, 0, 0);
    const std::array<qw::PackedWeight, 2> half{make_weight(kQType, kHalf, kHidden, kSeed, 0, 0),
                                               make_weight(kQType, kHalf, kHidden, kSeed, kHalf, 0)};
    // Distinct global ids in [0, kValidRows), shuffled so that the rows' order and the ids' differ.
    std::vector<std::int32_t> map(kValidRows);
    std::iota(map.begin(), map.end(), 0);
    std::mt19937 rng(kSeed);
    std::shuffle(map.begin(), map.end(), rng);
    map.resize(kRows);
    const std::array<std::vector<std::int32_t>, 2> half_map{
        std::vector<std::int32_t>(map.begin(), map.begin() + kHalf),
        std::vector<std::int32_t>(map.begin() + kHalf, map.end())};

    set_device(ec, 0);
    const RankWeight parent_weight = upload(parent);
    const DeviceBuffer map_device  = to_device(map);
    std::array<RankWeight, 2> half_weight;
    std::array<DeviceBuffer, 2> half_map_device;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        half_weight[rank]     = upload(half[rank]);
        half_map_device[rank] = to_device(half_map[rank]);
    }
    const Tensor map_t(map_device.p, DType::I32, {kRows});
    const std::array<Tensor, 2> half_map_t{Tensor(half_map_device[0].p, DType::I32, {kHalf}),
                                           Tensor(half_map_device[1].p, DType::I32, {kHalf})};

    // 7, 14, 21, 28: K=7 at C=1..4; 1, 16 and 17 the direct route's ends, 32/33/48 the M64 tiles.
    for (const std::int32_t columns : {1, 7, 14, 16, 17, 21, 28, 32, 33, 48}) {
        const std::string label = "q4 head U=" + std::to_string(columns);
        std::vector<float> activation(static_cast<std::size_t>(kHidden) * columns);
        fill_uniform(activation, kSeed * 31U + static_cast<std::uint32_t>(columns), -1.0F, 1.0F);
        round_to_bf16(activation);
        std::array<DeviceBuffer, 2> x_device;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            set_device(ec, static_cast<int>(rank));
            x_device[rank] = to_device_bf16(activation);
        }
        const std::array<Tensor, 2> x{Tensor(x_device[0].p, DType::BF16, {kHidden, columns}),
                                      Tensor(x_device[1].p, DType::BF16, {kHidden, columns})};

        // The whole head on rank 0.
        SplitBuffers::Result expected;
        {
            set_device(ec, 0);
            GuardedDeviceBuffer ids(static_cast<std::size_t>(columns) * kTopK * 4);
            GuardedDeviceBuffer scores(static_cast<std::size_t>(columns) * kTopK * 4);
            Tensor ids_t(ids.data(), DType::I32, {kTopK, columns});
            Tensor scores_t(scores.data(), DType::FP32, {kTopK, columns});
            WorkspaceArena workspace(ops::linear_topk_workspace_capacity_bytes(
                kQType, kRows, kHidden, columns, columns));
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear_topk(x[0], parent_weight.weight, map_t, ids_t, scores_t, workspace,
                             ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            const auto count = static_cast<std::size_t>(columns) * kTopK;
            expected = {from_device<std::int32_t>(ids.data(), count),
                        from_device<std::uint32_t>(scores.data(), count)};
        }
        std::array<std::optional<WorkspaceArena>, 2> arena;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            set_device(ec, static_cast<int>(rank));
            arena[rank].emplace(ops::linear_topk_split_workspace_capacity_bytes(
                kQType, kHalf, kHidden, columns, columns));
        }
        int from_rank1 = 0;
        {
            // How many of the whole head's winners sit in rank 1's block (the map's second half).
            std::vector<char> in_rank1(kValidRows, 0);
            for (const auto id : half_map[1]) { in_rank1[static_cast<std::size_t>(id)] = 1; }
            for (const auto id : expected.ids) { from_rank1 += in_rank1[static_cast<std::size_t>(id)]; }
        }
        std::cout << "  " << label << ": " << from_rank1 << "/" << columns * kTopK
                  << " winners from rank 1's block\n";
        failures += check_exchange(
            label, ec, mailbox, columns, expected, [&](SplitBuffers& b, const ops::PeerEvents*) {
                for (int rank = 0; rank < 2; ++rank) {
                    const auto r = static_cast<std::size_t>(rank);
                    set_device(ec, rank);
                    ops::linear_topk_split(x[r], half_weight[r].weight, half_map_t[r], b.ids_t[r],
                                           b.scores_t[r], *arena[r], ec.dev[r]->stream);
                }
            });
    }
    return failures;
}

// Planted local lists, one case per column: the merge against the host's sort of the 32 keys.
int run_planted(const ExecutionContext& ec, ops::PeerMailbox& mailbox) {
    struct Column {
        std::array<std::vector<std::pair<float, std::int32_t>>, 2> lists; // descending per rank
    };
    std::vector<Column> cases;
    const auto list = [](float top, float step, std::int32_t id0, std::int32_t id_step) {
        std::vector<std::pair<float, std::int32_t>> out;
        for (int i = 0; i < kTopK; ++i) { out.emplace_back(top - step * i, id0 + id_step * i); }
        return out;
    };
    cases.push_back({{list(9.0F, 0.5F, 100, 1), list(9.0F, 0.5F, 50, 1)}});   // equal scores, rank 1 ids lower
    cases.push_back({{list(9.0F, 0.5F, 50, 1), list(9.0F, 0.5F, 100, 1)}});   // equal scores, rank 0 ids lower
    cases.push_back({{list(1.0F, 0.1F, 7, 3), list(20.0F, 0.1F, 900, 3)}});   // all of rank 1 wins
    cases.push_back({{list(20.0F, 0.1F, 900, 3), list(1.0F, 0.1F, 7, 3)}});   // all of rank 0 wins
    cases.push_back({{list(0.0F, 0.25F, 10, 2), list(0.25F, 0.25F, 11, 2)}}); // interleaved around 0
    {
        Column c{{list(3.0F, 1.0F, 400, 1), list(3.0F, 1.0F, 300, 1)}};
        c.lists[0][3] = {-0.0F, 7};   // -0 and +0 are one score: the lower id first
        c.lists[1][3] = {0.0F, 6};
        c.lists[0][15] = {-1e30F, INT_MAX}; // the sentinel sorts below every finite key
        cases.push_back(c);
    }
    const auto columns = static_cast<std::int32_t>(cases.size());
    std::array<std::vector<std::int32_t>, 2> ids;
    std::array<std::vector<float>, 2> scores;
    SplitBuffers::Result expected;
    for (const auto& c : cases) {
        std::vector<std::uint64_t> keys;
        for (std::size_t r = 0; r < 2; ++r) {
            for (const auto& [score, id] : c.lists[r]) {
                ids[r].push_back(id);
                scores[r].push_back(score);
                keys.push_back(host_key(score, id));
            }
        }
        std::sort(keys.begin(), keys.end(), std::greater<>());
        for (int i = 0; i < kTopK; ++i) {
            const std::uint64_t key = keys[static_cast<std::size_t>(i)];
            const auto ordered      = static_cast<std::uint32_t>(key >> 32);
            const std::uint32_t bits =
                (ordered & 0x80000000U) != 0 ? (ordered ^ 0x80000000U) : ~ordered;
            expected.ids.push_back(key == 0 ? INT_MAX
                                            : static_cast<std::int32_t>(
                                                  0xffffffffU - static_cast<std::uint32_t>(key)));
            expected.score_bits.push_back(bits);
        }
    }
    std::array<DeviceBuffer, 2> ids_device, scores_device;
    for (std::size_t r = 0; r < 2; ++r) {
        set_device(ec, static_cast<int>(r));
        ids_device[r]    = to_device(ids[r]);
        scores_device[r] = to_device(scores[r]);
    }
    return check_exchange("planted", ec, mailbox, columns, expected,
                          [&](SplitBuffers& b, const ops::PeerEvents*) {
                              for (int rank = 0; rank < 2; ++rank) {
                                  const auto r = static_cast<std::size_t>(rank);
                                  set_device(ec, rank);
                                  cuda_check(cudaMemcpyAsync(b.ids_t[r].data, ids_device[r].p,
                                                             b.ids_t[r].bytes(),
                                                             cudaMemcpyDeviceToDevice,
                                                             ec.dev[r]->stream),
                                             "cudaMemcpyAsync");
                                  cuda_check(cudaMemcpyAsync(b.scores_t[r].data,
                                                             scores_device[r].p,
                                                             b.scores_t[r].bytes(),
                                                             cudaMemcpyDeviceToDevice,
                                                             ec.dev[r]->stream),
                                             "cudaMemcpyAsync");
                              }
                          });
}

int verify_registry() {
    try {
        if (ops::linear_topk_split_workspace_capacity_bytes(kQType, kHalf, kHidden, 1, 128) == 0) {
            std::cerr << "registry: the Q4 head block reported no workspace\n";
            return 1;
        }
        (void)ops::linear_topk_split_workspace_capacity_bytes(kQType, kRows, kHidden, 1, 1);
        std::cerr << "registry: the whole head was accepted as a block\n";
        return 1;
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& error) {
        std::cerr << "registry: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK registry: Q4 head block\n";
    return 0;
}

} // namespace

int main() {
    if (verify_registry() != 0) {
        std::cout << "FAIL proposal top-k split (registry)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: proposal top-k split requires two CUDA devices, found " << device_count
                  << '\n';
        return 77;
    }
    int failures = 0;
    try {
        const ExecutionContext ec({0, 1});
        std::cout << "peer access: " << (ops::enable_peer_access(ec) ? "direct" : "host-staged")
                  << '\n';
        ops::PeerMailbox mailbox(ec, 48 * ops::kTopKSplitCandidateRows * 2);
        failures += run_planted(ec, mailbox);
        failures += run_head(ec, mailbox);
    } catch (const std::exception& error) {
        std::cerr << "proposal top-k split: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " proposal top-k split\n";
    return failures ? 1 : 0;
}
