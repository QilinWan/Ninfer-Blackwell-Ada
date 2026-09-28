// Two-device vocabulary-split argmax of the optimized MTP proposal head.
//
// The Q4 proposal head `[131072,5120]` splits its rows, the proposal vocabulary, into two
// `[65536,5120]` halves. Each rank projects its half with linear_column_parallel, takes argmax() of
// its block and packs its candidate (argmax_split_pack); one allreduce_sum of the candidates gives
// rank 0 both, and argmax_split_select picks the complete row. The suite checks
//
//   1. every half row equals the whole head's row bit for bit (the half resolves the whole head's
//      kernels, which compute each row alone), at the decode and proposal column counts;
//   2. the selected row equals argmax() over the whole head's logits, eagerly (staged copies) and
//      in a replayed two-device graph through the mailbox;
//   3. the selection on planted logits: equal maxima across and inside the halves, the maximum in
//      either half, all -inf, and a NaN in row 0, at a tiled (65536) and a one-tile (300) block.
//
// The patterned Q4 fixture repeats a few hundred row classes, so the whole head's maximum recurs in
// both halves and case 2 exercises the tie across the halves (rank 0 wins every column); rank 1
// winning is case 3's.
//
// Every case needs two CUDA devices in one process and the suite reports 77 with fewer. The
// registry probe is host-only and runs first.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/peer_mailbox.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "core/weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/split_test_support.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr std::int32_t kRows   = 131072;
constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kHalf   = kRows / 2;
constexpr QType kQType         = QType::Q4_G64_FP16;

// Per-rank operands of the split argmax over `rows` rows per rank and `columns` columns.
struct SplitBuffers {
    std::array<std::optional<GuardedDeviceBuffer>, 2> local, candidates, staging;
    std::optional<GuardedDeviceBuffer> selected; // rank 0
    std::array<Tensor, 2> local_t, candidates_t, staging_t;
    Tensor selected_t;

    SplitBuffers(const ExecutionContext& ec, std::int32_t columns) {
        const auto count = static_cast<std::size_t>(columns);
        for (int rank = 0; rank < 2; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            local[r].emplace(count * sizeof(std::int32_t));
            candidates[r].emplace(count * ops::kArgmaxSplitCandidateRows * sizeof(std::uint16_t));
            staging[r].emplace(count * ops::kArgmaxSplitCandidateRows * sizeof(std::uint16_t));
            candidates[r]->fill(0x7f);
            local_t[r] = Tensor(local[r]->data(), DType::I32, {columns});
            candidates_t[r] = Tensor(candidates[r]->data(), DType::BF16,
                                     {ops::kArgmaxSplitCandidateRows, columns});
            staging_t[r] = Tensor(staging[r]->data(), DType::BF16,
                                  {ops::kArgmaxSplitCandidateRows, columns});
        }
        set_device(ec, 0);
        selected.emplace(count * sizeof(std::int32_t));
        selected->fill(0xcd);
        selected_t = Tensor(selected->data(), DType::I32, {columns});
    }

    int verify_guards(const ExecutionContext& ec, const std::string& label) {
        int failures = 0;
        for (int rank = 0; rank < 2; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            failures += local[r]->verify_guards(label + " local");
            failures += candidates[r]->verify_guards(label + " candidates");
            failures += staging[r]->verify_guards(label + " staging");
        }
        set_device(ec, 0);
        return failures + selected->verify_guards(label + " selected");
    }
};

// Both ranks' argmax and pack, the exchange and rank 0's selection, as
// execution::proposal_argmax_split issues them after the projection.
void issue_split(const ExecutionContext& ec, const ops::PeerEvents& events,
                 const std::array<Tensor, 2>& logits, SplitBuffers& b) {
    const std::int32_t rows = logits[0].ne[0];
    for (int rank = 0; rank < 2; ++rank) {
        const auto r = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        ops::argmax(logits[r], b.local_t[r], rows, ec.dev[r]->stream);
        ops::argmax_split_pack(logits[r], b.local_t[r], rank, b.candidates_t[r],
                               ec.dev[r]->stream);
    }
    ops::allreduce_sum(b.candidates_t, b.staging_t, ec, events);
    set_device(ec, 0);
    ops::argmax_split_select(b.candidates_t[0], rows, b.selected_t, ec.dev[0]->stream);
}

std::vector<int> selected_rows(const ExecutionContext& ec, SplitBuffers& b,
                               std::int32_t columns) {
    set_device(ec, 0);
    return from_device<int>(b.selected->data(), static_cast<std::size_t>(columns));
}

// The split selection eagerly (the staged path) and, when `mailbox` is given, replayed from one
// two-device graph through it; both must equal `expected`.
int check_selection(const std::string& label, const ExecutionContext& ec,
                    const std::array<Tensor, 2>& logits, const std::vector<int>& expected,
                    ops::PeerMailbox* mailbox) {
    const std::int32_t columns = logits[0].ne[1];
    int failures               = 0;
    {
        const ops::PeerEvents events(ec);
        SplitBuffers b(ec, columns);
        retire_staging(ec);
        issue_split(ec, events, logits, b);
        synchronize_both(ec);
        failures += verify_exact((label + " eager").c_str(), selected_rows(ec, b, columns),
                                 expected);
        failures += b.verify_guards(ec, label + " eager");
    }
    if (mailbox == nullptr) { return failures; }
    ops::PeerEvents events(ec);
    events.attach_mailbox(mailbox);
    SplitBuffers b(ec, columns);
    retire_staging(ec);
    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    set_device(ec, 0);
    definition.capture(
        ec.dev[0]->stream,
        [&] {
            issue_split(ec, events, logits, b);
            set_device(ec, 0);
        },
        DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    for (int replay = 0; replay < 3; ++replay) {
        set_device(ec, 0);
        b.selected->fill(0xcd);
        retire_staging(ec);
        executable.launch(ec.dev[0]->stream);
        synchronize_both(ec);
        failures += verify_exact((label + " mailbox replay " + std::to_string(replay)).c_str(),
                                 selected_rows(ec, b, columns), expected);
    }
    if (mailbox->hang_reported()) {
        std::cerr << label << ": the mailbox reported a hang\n";
        ++failures;
    }
    return failures + b.verify_guards(ec, label + " mailbox");
}

std::vector<int> reference_argmax(const ExecutionContext& ec, const Tensor& logits,
                                  std::int32_t valid_rows) {
    set_device(ec, 0);
    const std::int32_t columns = logits.ne[1];
    GuardedDeviceBuffer out(static_cast<std::size_t>(columns) * sizeof(std::int32_t));
    Tensor out_t(out.data(), DType::I32, {columns});
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ops::argmax(logits, out_t, valid_rows, ec.dev[0]->stream);
    cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
    return from_device<int>(out.data(), static_cast<std::size_t>(columns));
}

int run_head(const ExecutionContext& ec, ops::PeerMailbox& mailbox) {
    constexpr std::uint32_t kSeed = 409U;
    int failures                  = 0;
    const qw::PackedWeight parent = make_weight(kQType, kRows, kHidden, kSeed, 0, 0);
    const std::array<qw::PackedWeight, 2> half{make_weight(kQType, kHalf, kHidden, kSeed, 0, 0),
                                               make_weight(kQType, kHalf, kHidden, kSeed, kHalf, 0)};
    for (const std::int32_t row : {0, 1, 4095, kHalf - 1}) {
        for (const std::int32_t column : {0, 63, 64, kHidden - 1}) {
            for (std::size_t rank = 0; rank < 2; ++rank) {
                if (qw::logical_weight_fp64(half[rank], row, column) !=
                    qw::logical_weight_fp64(parent, row + static_cast<std::int32_t>(rank) * kHalf,
                                            column)) {
                    std::cerr << "proposal head half " << rank << " is not the parent's block\n";
                    return 1;
                }
            }
        }
    }

    set_device(ec, 0);
    const RankWeight parent_weight = upload(parent);
    std::array<RankWeight, 2> half_weight;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        half_weight[rank] = upload(half[rank]);
    }

    // T=1 is C=1 decode (GEMV); 2..8 the sliced-K routes of C=2..8; 9, 17 and 33 later routes.
    for (const std::int32_t tokens : {1, 2, 4, 5, 8, 9, 17, 33}) {
        const std::string label = "q4 proposal head T=" + std::to_string(tokens);
        std::vector<float> activation(static_cast<std::size_t>(kHidden) * tokens);
        fill_uniform(activation, kSeed * 37U + static_cast<std::uint32_t>(tokens), -1.0F, 1.0F);
        round_to_bf16(activation);
        std::array<DeviceBuffer, 2> x_device;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            set_device(ec, static_cast<int>(rank));
            x_device[rank] = to_device_bf16(activation);
        }
        const std::size_t whole_elements = static_cast<std::size_t>(kRows) * tokens;
        const std::size_t half_elements  = static_cast<std::size_t>(kHalf) * tokens;

        set_device(ec, 0);
        GuardedDeviceBuffer whole(whole_elements * sizeof(std::uint16_t));
        DeviceArena whole_arena(1);
        Tensor whole_t(whole.data(), DType::BF16, {kRows, tokens});
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::linear(Tensor(x_device[0].p, DType::BF16, {kHidden, tokens}), parent_weight.weight,
                    whole_t, ops::LinearPolicy::A16Only, whole_arena, ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
        const auto whole_bits = from_device<std::uint16_t>(whole.data(), whole_elements);
        const std::vector<int> expected = reference_argmax(ec, whole_t, kRows);

        std::array<std::optional<GuardedDeviceBuffer>, 2> partial;
        std::array<std::optional<DeviceArena>, 2> arena;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            set_device(ec, static_cast<int>(rank));
            partial[rank].emplace(half_elements * sizeof(std::uint16_t));
            partial[rank]->fill(0xff);
            arena[rank].emplace(1);
        }
        const std::array<Tensor, 2> x{Tensor(x_device[0].p, DType::BF16, {kHidden, tokens}),
                                      Tensor(x_device[1].p, DType::BF16, {kHidden, tokens})};
        const std::array<Tensor, 2> logits{
            Tensor(partial[0]->data(), DType::BF16, {kHalf, tokens}),
            Tensor(partial[1]->data(), DType::BF16, {kHalf, tokens})};
        retire_staging(ec);
        ops::linear_column_parallel(x, {half_weight[0].weight, half_weight[1].weight}, logits,
                                    ops::LinearPolicy::A16Only, {&*arena[0], &*arena[1]}, ec);
        synchronize_both(ec);
        for (std::size_t rank = 0; rank < 2; ++rank) {
            set_device(ec, static_cast<int>(rank));
            failures += partial[rank]->verify_guards(label + " half " + std::to_string(rank));
            const auto half_bits = from_device<std::uint16_t>(partial[rank]->data(), half_elements);
            std::vector<std::uint16_t> block(half_elements);
            for (std::int32_t token = 0; token < tokens; ++token) {
                const auto source = static_cast<std::size_t>(token) * kRows + rank * kHalf;
                std::copy(whole_bits.begin() + static_cast<std::ptrdiff_t>(source),
                          whole_bits.begin() + static_cast<std::ptrdiff_t>(source + kHalf),
                          block.begin() + static_cast<std::ptrdiff_t>(token) * kHalf);
            }
            failures += verify_exact(
                (label + " half " + std::to_string(rank) + " rows equal the whole head's").c_str(),
                half_bits, block);
        }
        int in_rank1 = 0;
        for (const int row : expected) { in_rank1 += row >= kHalf ? 1 : 0; }
        std::cout << "  " << label << ": argmax in rank 1's half for " << in_rank1 << "/"
                  << tokens << " columns\n";
        failures += check_selection(label, ec, logits, expected, &mailbox);
    }
    return failures;
}

// Planted logits, one case per column, against argmax() over the concatenated block and against
// the row each case names.
int run_planted(const ExecutionContext& ec, ops::PeerMailbox& mailbox, std::int32_t rows) {
    const std::string label = "planted R=" + std::to_string(rows);
    const float inf         = std::numeric_limits<float>::infinity();
    const float nan         = std::numeric_limits<float>::quiet_NaN();
    const std::int32_t a = rows / 3, b = rows - 2; // a row in each half: a and rows + b
    struct Case {
        std::vector<std::pair<std::int32_t, float>> planted; // complete row, value
        bool all_negative_infinity;
        int expected;
    };
    const std::vector<Case> cases{
        {{{a, 9.0F}, {rows + b, 9.0F}}, false, a},              // tie across the halves
        {{{a, 8.0F}, {rows + b, 9.0F}}, false, rows + b},       // maximum in rank 1's half
        {{{a, 9.0F}, {rows + b, 8.5F}}, false, a},              // maximum in rank 0's half
        {{{rows + b, 9.0F}, {rows + 1, 9.0F}}, false, rows + 1}, // tie inside rank 1's half
        {{{rows, 9.0F}, {rows - 1, 9.0F}}, false, rows - 1},    // tie at the boundary
        {{}, true, 0},                                          // all -inf
        {{{0, nan}, {rows + b, 9.0F}}, false, 0},               // NaN in row 0
    };
    const auto columns = static_cast<std::int32_t>(cases.size());
    std::vector<float> values(static_cast<std::size_t>(2 * rows) * columns);
    fill_uniform(values, 0x5eedU + static_cast<std::uint32_t>(rows), -4.0F, 4.0F);
    for (std::int32_t c = 0; c < columns; ++c) {
        const auto base = static_cast<std::size_t>(c) * 2 * rows;
        if (cases[static_cast<std::size_t>(c)].all_negative_infinity) {
            std::fill(values.begin() + static_cast<std::ptrdiff_t>(base),
                      values.begin() + static_cast<std::ptrdiff_t>(base + 2 * rows), -inf);
        }
        for (const auto& [row, value] : cases[static_cast<std::size_t>(c)].planted) {
            values[base + static_cast<std::size_t>(row)] = value;
        }
    }
    round_to_bf16(values);
    std::vector<int> expected;
    for (const auto& c : cases) { expected.push_back(c.expected); }

    set_device(ec, 0);
    const DeviceBuffer whole = to_device_bf16(values);
    const std::vector<int> reference =
        reference_argmax(ec, Tensor(whole.p, DType::BF16, {2 * rows, columns}), 2 * rows);
    int failures = verify_exact((label + " argmax() names the planted rows").c_str(), reference,
                                expected);
    std::array<std::vector<float>, 2> blocks;
    std::array<DeviceBuffer, 2> block_device;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        for (std::int32_t c = 0; c < columns; ++c) {
            const auto source = static_cast<std::size_t>(c) * 2 * rows + rank * rows;
            blocks[rank].insert(blocks[rank].end(),
                                values.begin() + static_cast<std::ptrdiff_t>(source),
                                values.begin() + static_cast<std::ptrdiff_t>(source + rows));
        }
        set_device(ec, static_cast<int>(rank));
        block_device[rank] = to_device_bf16(blocks[rank]);
    }
    const std::array<Tensor, 2> logits{Tensor(block_device[0].p, DType::BF16, {rows, columns}),
                                       Tensor(block_device[1].p, DType::BF16, {rows, columns})};
    return failures + check_selection(label, ec, logits, reference, &mailbox);
}

int verify_registry() {
    try {
        if (ops::linear_workspace_capacity_bytes(kQType, kHalf, kHidden,
                                                 ops::LinearPolicy::A16Only, 1, 2048) != 0) {
            std::cerr << "registry: the Q4 proposal head half reported workspace\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "registry: the Q4 proposal head half was rejected: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK registry: Q4 proposal head half\n";
    return 0;
}

} // namespace

int main() {
    if (verify_registry() != 0) {
        std::cout << "FAIL proposal head split (registry)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: proposal head split requires two CUDA devices, found " << device_count
                  << '\n';
        return 77;
    }
    int failures = 0;
    try {
        const ExecutionContext ec({0, 1});
        std::cout << "peer access: " << (ops::enable_peer_access(ec) ? "direct" : "host-staged")
                  << '\n';
        ops::PeerMailbox mailbox(ec, 4096);
        failures += run_planted(ec, mailbox, kHalf);
        failures += run_planted(ec, mailbox, 300);
        failures += run_head(ec, mailbox);
    } catch (const std::exception& error) {
        std::cerr << "proposal head split: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " proposal head split\n";
    return failures ? 1 : 0;
}
