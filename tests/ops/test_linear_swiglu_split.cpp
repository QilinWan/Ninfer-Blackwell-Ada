// Two-device parity of linear_swiglu_column_parallel, and of the MLP it feeds:
// linear_swiglu_column_parallel followed by linear_add_row_parallel.
//
// The per-format LinearSwiGLU suites qualify the [17408,5120] half against the FP64 oracle. This
// suite checks that splitting the gate/up projection across two devices reproduces the
// single-device result: each case runs linear_swiglu() on device 0 over the whole [34816,5120]
// weight and the split form over the two shards, then compares rank r's output with rows
// [r*8704,(r+1)*8704) of the whole result.
//
// A gate/up shard is not one block of its parent. Rank r's shard stacks the parent's gate rows
// [r*8704,(r+1)*8704) on its up rows [17408+r*8704,17408+(r+1)*8704), so that it is itself a
// gate/up weight with M = 8704. The fixture keys every code and scale on the global coordinate, so
// each block is generated at its parent origin and the two are stacked; every case first checks,
// through the fixture's independent decoder, that each shard row holds the parent row it stands
// for and that the blocks and shards differ, so a generator defect fails as a weight mismatch
// rather than passing as a comparison of a block with a copy of itself.
//
// Column-parallel ranks evaluate the dot products the whole-weight kernel evaluates for their
// rows, so their result matches to rounding. The MLP cases chain the split output straight into
// the row-parallel down projection and compare with linear_swiglu() plus linear_add() on one
// device; the row split adds two BF16 roundings of partials, so that bound is stated against the
// largest output.
//
// Every case needs two CUDA devices in one process and the suite reports 77 with fewer. The
// registry probe is host-only and runs first.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include "core/device.h"
#include "core/weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/split_test_support.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr std::int32_t kGateUpRows      = 34816;
constexpr std::int32_t kInputRows       = 5120;
constexpr std::int32_t kIntermediate    = kGateUpRows / 2;   // 17408
constexpr std::int32_t kShardGateUpRows = kGateUpRows / 2;   // 17408
constexpr std::int32_t kShardHalf       = kIntermediate / 2; // 8704
constexpr std::int32_t kHiddenRows      = 5120;

// Stacks two NVFP4 or FP8 blocks of one K into a standalone tensor holding `top`'s rows followed by
// `bottom`'s. Every plane of both formats is addressed by whole rows (the NVFP4 scale plane by
// 128-row tiles, which both block heights are multiples of), so the planes concatenate.
qw::PackedWeight stack_rows(const qw::PackedWeight& top, const qw::PackedWeight& bottom) {
    const Weight& upper = top.weight;
    const Weight& lower = bottom.weight;
    const bool nvfp4    = upper.qtype == QType::NVFP4;
    if (upper.qtype != lower.qtype || upper.layout != lower.layout || upper.k != lower.k ||
        (!nvfp4 && upper.qtype != QType::FP8_E4M3FN_ROW_BF16) || top.high_plane_bytes != 0 ||
        bottom.high_plane_bytes != 0 || (nvfp4 && (upper.n % 128) != 0) ||
        (nvfp4 && upper.weight_scale_divisor != lower.weight_scale_divisor)) {
        throw std::invalid_argument("stack_rows: blocks do not stack");
    }

    qw::PackedWeight stacked;
    stacked.code_plane_bytes = top.code_plane_bytes + bottom.code_plane_bytes;
    stacked.scale_plane_offset =
        qw::detail::align_up_size(static_cast<std::size_t>(stacked.code_plane_bytes), 256);
    stacked.scale_plane_bytes = top.scale_plane_bytes + bottom.scale_plane_bytes;
    std::size_t total =
        static_cast<std::size_t>(stacked.scale_plane_offset + stacked.scale_plane_bytes);
    if (nvfp4) {
        stacked.weight_divisor_offset = total;
        total += sizeof(float);
    }
    stacked.payload.assign(total, 0);

    const auto copy = [&](std::uint64_t destination, const qw::PackedWeight& block,
                          std::uint64_t source, std::uint64_t bytes) {
        std::memcpy(stacked.payload.data() + destination, block.payload.data() + source, bytes);
    };
    copy(0, top, 0, top.code_plane_bytes);
    copy(top.code_plane_bytes, bottom, 0, bottom.code_plane_bytes);
    copy(stacked.scale_plane_offset, top, top.scale_plane_offset, top.scale_plane_bytes);
    copy(stacked.scale_plane_offset + top.scale_plane_bytes, bottom, bottom.scale_plane_offset,
         bottom.scale_plane_bytes);
    if (nvfp4) {
        copy(stacked.weight_divisor_offset, top, top.weight_divisor_offset, sizeof(float));
    }

    Weight& weight         = stacked.weight;
    weight                 = upper;
    weight.n               = upper.n + lower.n;
    weight.shape[0]        = weight.n;
    weight.padded_shape[0] = weight.n;
    weight.payload         = stacked.payload.data();
    weight.payload_bytes   = stacked.payload.size();
    weight.qdata           = stacked.payload.data();
    weight.scales          = stacked.payload.data() + stacked.scale_plane_offset;
    if (!nvfp4) {
        // One BF16 multiplier per output row.
        weight.scale_ne[0] = weight.n;
        weight.scale_nb[1] = static_cast<std::int64_t>(weight.n) * 2;
        weight.scale_nb[2] = weight.scale_nb[1];
        weight.scale_nb[3] = weight.scale_nb[1];
    }
    return stacked;
}

struct GateUpShard {
    qw::PackedWeight gate;
    qw::PackedWeight up;
    qw::PackedWeight shard;
};

// Rank r's shard of the [2M,K] gate/up parent split evenly: gate rows [r*M/2,(r+1)*M/2) stacked on
// up rows [M+r*M/2,M+(r+1)*M/2).
GateUpShard make_gate_up_shard(QType qtype, std::uint32_t seed, int rank) {
    GateUpShard result{
        make_weight(qtype, kShardHalf, kInputRows, seed, rank * kShardHalf, 0),
        make_weight(qtype, kShardHalf, kInputRows, seed, kIntermediate + rank * kShardHalf, 0),
        {}};
    result.shard = stack_rows(result.gate, result.up);
    return result;
}

// Coordinates straddling the NVFP4 32- and 128-row scale tiles, the quantization groups, and the
// extent's own first and last index.
std::vector<std::int32_t> seam_samples(std::int32_t extent) {
    const std::vector<std::int32_t> probes{0,   1,   31,  32,  33,  63,         64,        127,
                                           128, 129, 255, 256, 511, extent / 2, extent - 1};
    std::vector<std::int32_t> result;
    for (const std::int32_t probe : probes) {
        if (probe >= 0 && probe < extent &&
            std::find(result.begin(), result.end(), probe) == result.end()) {
            result.push_back(probe);
        }
    }
    return result;
}

// Checks that each row of rank r's shard is the parent row it stands for, on both halves.
int verify_gate_up_shard(const std::string& label, const qw::PackedWeight& parent,
                         const qw::PackedWeight& shard, int rank) {
    for (const std::int32_t row : seam_samples(kShardHalf)) {
        for (const std::int32_t column : seam_samples(kInputRows)) {
            const std::array<std::array<std::int32_t, 2>, 2> pairs{{
                {row, rank * kShardHalf + row},
                {kShardHalf + row, kIntermediate + rank * kShardHalf + row},
            }};
            for (const auto& [shard_row, parent_row] : pairs) {
                const double got      = qw::logical_weight_fp64(shard, shard_row, column);
                const double expected = qw::logical_weight_fp64(parent, parent_row, column);
                if (got != expected) {
                    std::cerr << label << ": shard (" << shard_row << ',' << column
                              << ") is not the parent's (" << parent_row << ',' << column
                              << "): " << got << " vs " << expected << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_distinct(const std::string& label, const std::vector<std::uint8_t>& first,
                    const std::vector<std::uint8_t>& second) {
    if (first != second) { return 0; }
    std::cerr << label << ": the two payloads are byte-identical\n";
    return 1;
}

// Builds and checks the parent and both shards of one gate/up weight, then uploads the parent to
// device 0 and shard r to device r.
struct GateUpWeights {
    RankWeight parent;
    std::array<RankWeight, 2> shard;
};

int make_gate_up_weights(const std::string& label, QType qtype, std::uint32_t seed,
                         const ExecutionContext& ec, std::optional<GateUpWeights>& out) {
    const qw::PackedWeight parent = make_weight(qtype, kGateUpRows, kInputRows, seed, 0, 0);
    std::array<GateUpShard, 2> shard{make_gate_up_shard(qtype, seed, 0),
                                     make_gate_up_shard(qtype, seed, 1)};
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot         = static_cast<std::size_t>(rank);
        const std::string where = label + " shard " + std::to_string(rank);
        failures +=
            verify_distinct(where + " gate/up", shard[slot].gate.payload, shard[slot].up.payload);
        failures += verify_gate_up_shard(where, parent, shard[slot].shard, rank);
    }
    failures += verify_distinct(label + " shards", shard[0].shard.payload, shard[1].shard.payload);
    if (failures != 0) { return failures; }

    out.emplace();
    set_device(ec, 0);
    out->parent = upload(parent);
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        out->shard[static_cast<std::size_t>(rank)] =
            upload(shard[static_cast<std::size_t>(rank)].shard);
    }
    return 0;
}

std::size_t swiglu_workspace_bytes(QType qtype, std::int32_t gate_up_rows, ops::LinearPolicy policy,
                                   std::int32_t tokens) {
    return std::max<std::size_t>(ops::linear_swiglu_workspace_capacity_bytes(
                                     qtype, gate_up_rows, kInputRows, policy, tokens, tokens),
                                 1);
}

// 2u of the largest output (one to two BF16 ulp, see kBf16UnitRoundoff) and of relative L2. A
// column rank evaluates the whole-weight dot products of its rows, and a row split adds two BF16
// roundings of partials comparable to the result, so about one ulp is the expected difference
// rather than the limit.
constexpr ReductionCriterion kSplitCriterion{2.0 * kBf16UnitRoundoff, 0.0, 2.0 * kBf16UnitRoundoff};

std::vector<float> make_activation(std::int32_t rows, std::int32_t tokens, std::uint32_t seed) {
    std::vector<float> activation(static_cast<std::size_t>(rows) * tokens);
    fill_uniform(activation, seed, -1.0F, 1.0F);
    round_to_bf16(activation);
    return activation;
}

struct Case {
    const char* label;
    QType qtype;
    std::uint32_t seed;
    std::vector<std::int32_t> tokens;
    std::vector<ops::LinearPolicy> policies;
};

int run_case(const Case& test_case, const ExecutionContext& ec) {
    const std::string head = test_case.label;
    std::cout << head << " [" << kGateUpRows << ',' << kInputRows << "] -> [" << kShardGateUpRows
              << ',' << kInputRows << "]\n";
    std::optional<GateUpWeights> weights;
    int failures = make_gate_up_weights(head, test_case.qtype, test_case.seed, ec, weights);
    if (failures != 0) { return failures; }

    for (const std::int32_t tokens : test_case.tokens) {
        const std::vector<float> activation = make_activation(
            kInputRows, tokens, test_case.seed * 31U + static_cast<std::uint32_t>(tokens));
        set_device(ec, 0);
        const DeviceBuffer parent_x = to_device_bf16(activation);
        std::array<DeviceBuffer, 2> shard_x;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            set_device(ec, static_cast<int>(rank));
            shard_x[rank] = to_device_bf16(activation);
        }

        for (const ops::LinearPolicy policy : test_case.policies) {
            const std::string label =
                head + " T=" + std::to_string(tokens) + " " + policy_name(policy);

            const std::size_t parent_elements = static_cast<std::size_t>(kIntermediate) * tokens;
            set_device(ec, 0);
            GuardedDeviceBuffer reference(parent_elements * sizeof(std::uint16_t));
            reference.fill(0xff);
            DeviceArena reference_arena(
                swiglu_workspace_bytes(test_case.qtype, kGateUpRows, policy, tokens));
            const Tensor reference_x(parent_x.p, DType::BF16, {kInputRows, tokens});
            Tensor reference_out(reference.data(), DType::BF16, {kIntermediate, tokens});
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear_swiglu(reference_x, weights->parent.weight, reference_out, policy,
                               reference_arena, ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += reference.verify_guards(label + " reference");
            const std::vector<double> expected =
                from_device_bf16(reference.data(), parent_elements);

            const std::size_t shard_elements = static_cast<std::size_t>(kShardHalf) * tokens;
            std::array<std::optional<GuardedDeviceBuffer>, 2> shard_out;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                set_device(ec, static_cast<int>(rank));
                shard_out[rank].emplace(shard_elements * sizeof(std::uint16_t));
                shard_out[rank]->fill(0xff);
                arena[rank].emplace(
                    swiglu_workspace_bytes(test_case.qtype, kShardGateUpRows, policy, tokens));
            }
            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kInputRows, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {kInputRows, tokens})};
            const std::array<Weight, 2> w{weights->shard[0].weight, weights->shard[1].weight};
            const std::array<Tensor, 2> out{
                Tensor(shard_out[0]->data(), DType::BF16, {kShardHalf, tokens}),
                Tensor(shard_out[1]->data(), DType::BF16, {kShardHalf, tokens})};
            const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::linear_swiglu_column_parallel(x, w, out, policy, workspace, ec);
            synchronize_both(ec);

            std::array<std::vector<double>, 2> observed;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                const std::string rank_label = label + " rank " + std::to_string(rank);
                set_device(ec, static_cast<int>(rank));
                failures += shard_out[rank]->verify_guards(rank_label);
                observed[rank] = from_device_bf16(shard_out[rank]->data(), shard_elements);
                // Rank r owns intermediate rows [r*kShardHalf, (r+1)*kShardHalf) of every token.
                std::vector<double> block(shard_elements);
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const std::size_t source = static_cast<std::size_t>(token) * kIntermediate +
                                               rank * static_cast<std::size_t>(kShardHalf);
                    std::copy(expected.begin() + static_cast<std::ptrdiff_t>(source),
                              expected.begin() + static_cast<std::ptrdiff_t>(source + kShardHalf),
                              block.begin() + static_cast<std::ptrdiff_t>(token) * kShardHalf);
                }
                failures += compare(rank_label, observed[rank], block, kSplitCriterion);
            }
            if (observed[0] == observed[1]) {
                std::cerr << label << ": both ranks produced the same output block\n";
                ++failures;
            }
        }
    }
    return failures;
}

// The MLP as the model composes it: gate/up + SwiGLU, then the down projection added into the
// residual. The split output of rank r is rank r's activation block of the row-parallel down
// projection, with no data movement in between.
struct MlpCase {
    const char* label;
    QType qtype;
    std::uint32_t seed;
    std::int32_t tokens;
    ops::LinearPolicy policy;
};

int run_mlp_case(const MlpCase& test_case, const ExecutionContext& ec,
                 const ops::PeerEvents& events) {
    const std::string head         = std::string(test_case.label) + " mlp";
    const std::int32_t tokens      = test_case.tokens;
    const QType qtype              = test_case.qtype;
    const ops::LinearPolicy policy = test_case.policy;
    const std::string label = head + " T=" + std::to_string(tokens) + " " + policy_name(policy);
    std::cout << label << '\n';

    std::optional<GateUpWeights> gate_up;
    int failures = make_gate_up_weights(head + " gate_up", qtype, test_case.seed, ec, gate_up);
    const qw::PackedWeight down_parent =
        make_weight(qtype, kHiddenRows, kIntermediate, test_case.seed + 1U, 0, 0);
    std::array<std::optional<qw::PackedWeight>, 2> down_shard;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        down_shard[rank].emplace(make_weight(qtype, kHiddenRows, kShardHalf, test_case.seed + 1U, 0,
                                             static_cast<std::int32_t>(rank) * kShardHalf));
    }
    failures +=
        verify_distinct(head + " down shards", down_shard[0]->payload, down_shard[1]->payload);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    const RankWeight down_parent_device = upload(down_parent);
    std::array<RankWeight, 2> down_device;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        down_device[rank] = upload(*down_shard[rank]);
    }

    const std::vector<float> activation =
        make_activation(kInputRows, tokens, test_case.seed * 31U + 7U);
    std::vector<float> residual_values(static_cast<std::size_t>(kHiddenRows) * tokens);
    fill_uniform(residual_values, test_case.seed * 97U + 11U, -2.0F, 2.0F);
    std::vector<std::uint16_t> residual_bits(residual_values.size());
    std::transform(residual_values.begin(), residual_values.end(), residual_bits.begin(),
                   f32_to_bf16);
    const std::size_t hidden_elements = residual_bits.size();
    const std::size_t hidden_bytes    = hidden_elements * sizeof(std::uint16_t);

    // Single-device reference on device 0.
    set_device(ec, 0);
    const DeviceBuffer parent_x = to_device_bf16(activation);
    DeviceBuffer parent_intermediate(static_cast<std::size_t>(kIntermediate) * tokens *
                                     sizeof(std::uint16_t));
    GuardedDeviceBuffer reference(hidden_bytes);
    reference.copy_from_host(residual_bits.data(), hidden_bytes);
    DeviceArena reference_arena(std::max(
        swiglu_workspace_bytes(qtype, kGateUpRows, policy, tokens),
        std::max<std::size_t>(ops::linear_add_workspace_capacity_bytes(
                                  qtype, kHiddenRows, kIntermediate, policy, tokens, tokens),
                              1)));
    {
        const Tensor x(parent_x.p, DType::BF16, {kInputRows, tokens});
        Tensor intermediate(parent_intermediate.p, DType::BF16, {kIntermediate, tokens});
        Tensor residual(reference.data(), DType::BF16, {kHiddenRows, tokens});
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::linear_swiglu(x, gate_up->parent.weight, intermediate, policy, reference_arena,
                           ec.dev[0]->stream);
        ops::linear_add(intermediate, down_parent_device.weight, residual, policy, reference_arena,
                        ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
    }
    failures += reference.verify_guards(label + " reference");
    const std::vector<double> expected = from_device_bf16(reference.data(), hidden_elements);

    // Split pipeline.
    std::array<DeviceBuffer, 2> shard_x;
    std::array<std::optional<DeviceBuffer>, 2> intermediate;
    std::array<std::optional<GuardedDeviceBuffer>, 2> residual;
    std::array<std::optional<DeviceBuffer>, 2> staging;
    std::array<std::optional<DeviceArena>, 2> arena;
    const std::size_t arena_bytes =
        std::max(swiglu_workspace_bytes(qtype, kShardGateUpRows, policy, tokens),
                 std::max<std::size_t>(ops::linear_add_row_parallel_workspace_capacity_bytes(
                                           qtype, kHiddenRows, kShardHalf, policy, tokens, tokens),
                                       1));
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        shard_x[rank] = to_device_bf16(activation);
        intermediate[rank].emplace(static_cast<std::size_t>(kShardHalf) * tokens *
                                   sizeof(std::uint16_t));
        residual[rank].emplace(hidden_bytes);
        residual[rank]->copy_from_host(residual_bits.data(), hidden_bytes);
        staging[rank].emplace(hidden_bytes);
        arena[rank].emplace(arena_bytes);
    }
    const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kInputRows, tokens}),
                                  Tensor(shard_x[1].p, DType::BF16, {kInputRows, tokens})};
    const std::array<Tensor, 2> activation_block{
        Tensor(intermediate[0]->p, DType::BF16, {kShardHalf, tokens}),
        Tensor(intermediate[1]->p, DType::BF16, {kShardHalf, tokens})};
    const std::array<Tensor, 2> residual_view{
        Tensor(residual[0]->data(), DType::BF16, {kHiddenRows, tokens}),
        Tensor(residual[1]->data(), DType::BF16, {kHiddenRows, tokens})};
    const std::array<Tensor, 2> staging_view{
        Tensor(staging[0]->p, DType::BF16, {kHiddenRows, tokens}),
        Tensor(staging[1]->p, DType::BF16, {kHiddenRows, tokens})};
    const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

    retire_staging(ec);
    ops::linear_swiglu_column_parallel(x, {gate_up->shard[0].weight, gate_up->shard[1].weight},
                                       activation_block, policy, workspace, ec);
    ops::linear_add_row_parallel(activation_block, {down_device[0].weight, down_device[1].weight},
                                 residual_view, staging_view, policy, workspace, ec, events);
    synchronize_both(ec);

    std::array<std::vector<double>, 2> observed;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        const std::string rank_label = label + " rank " + std::to_string(rank);
        set_device(ec, static_cast<int>(rank));
        failures += residual[rank]->verify_guards(rank_label);
        observed[rank] = from_device_bf16(residual[rank]->data(), hidden_elements);
        failures += compare(rank_label, observed[rank], expected, kSplitCriterion);
    }
    if (observed[0] != observed[1]) {
        std::cerr << label << ": the ranks disagree after the all-reduce\n";
        ++failures;
    }
    return failures;
}

// linear_swiglu_workspace_capacity_bytes() runs each format's shape resolver without a device, so
// the registration of the half is checked even where the parity cases must skip.
int verify_registry() {
    struct Entry {
        QType qtype;
        ops::LinearPolicy policy;
        std::vector<std::int32_t> tokens;
    };

    const std::vector<Entry> admitted{
        {QType::NVFP4, ops::LinearPolicy::A16Only, {1, 2, 16}},
        {QType::NVFP4, ops::LinearPolicy::AllowA4, {1, 2, 16, 48, 129, 1024}},
        {QType::FP8_E4M3FN_ROW_BF16, ops::LinearPolicy::A16Only, {1, 2, 48, 1024}},
        {QType::FP8_E4M3FN_ROW_BF16, ops::LinearPolicy::AllowA8, {1, 2, 48, 1024}},
    };
    int failures = 0;
    for (const Entry& entry : admitted) {
        for (const std::int32_t tokens : entry.tokens) {
            try {
                (void)ops::linear_swiglu_workspace_capacity_bytes(
                    entry.qtype, kShardGateUpRows, kInputRows, entry.policy, tokens, tokens);
            } catch (const std::exception& error) {
                std::cerr << "registry: qtype " << static_cast<int>(entry.qtype) << ' '
                          << policy_name(entry.policy) << " T=" << tokens
                          << " rejected the half: " << error.what() << '\n';
                ++failures;
            }
        }
    }

    struct Rejected {
        QType qtype;
        std::int32_t gate_up_rows;
    };

    // The groupwise gate/up problems register no half, and no other height is a gate/up problem.
    const std::vector<Rejected> rejected{
        {QType::Q4_G64_FP16, kShardGateUpRows},
        {QType::Q8_G32_FP16, kShardGateUpRows},
        {QType::NVFP4, 8704},
        {QType::FP8_E4M3FN_ROW_BF16, 8704},
    };
    for (const Rejected& entry : rejected) {
        try {
            (void)ops::linear_swiglu_workspace_capacity_bytes(
                entry.qtype, entry.gate_up_rows, kInputRows, ops::LinearPolicy::A16Only, 1, 1);
            std::cerr << "registry: [" << entry.gate_up_rows << ',' << kInputRows << "] qtype "
                      << static_cast<int>(entry.qtype) << " was admitted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    std::cout << (failures ? "FAIL" : "OK") << " registry\n";
    return failures;
}

int verify_split_rejections(const ExecutionContext& ec) {
    int failures            = 0;
    const auto expect_throw = [&](const char* what, auto&& body) {
        try {
            body();
        } catch (const std::invalid_argument&) { return; } catch (const std::exception& error) {
            std::cerr << "split rejection " << what << ": wrong exception: " << error.what()
                      << '\n';
            ++failures;
            return;
        }
        std::cerr << "split rejection " << what << ": accepted an invalid pair\n";
        ++failures;
    };

    const std::size_t x_bytes   = static_cast<std::size_t>(kInputRows) * 2 * sizeof(std::uint16_t);
    const std::size_t out_bytes = static_cast<std::size_t>(kShardHalf) * 2 * sizeof(std::uint16_t);
    set_device(ec, 0);
    DeviceBuffer x0(x_bytes);
    DeviceBuffer out0(out_bytes);
    set_device(ec, 1);
    DeviceBuffer x1(x_bytes);
    DeviceBuffer out1(out_bytes);

    Weight weight{};
    weight.qtype = QType::FP8_E4M3FN_ROW_BF16;
    weight.n     = kShardGateUpRows;
    weight.k     = kInputRows;

    expect_throw("token count", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 2}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kShardHalf, 2}),
                                        Tensor(out1.p, DType::BF16, {kShardHalf, 1})};
        ops::linear_swiglu_column_parallel(x, {weight, weight}, out, ec);
    });
    expect_throw("column K", [&] {
        Weight other = weight;
        other.k      = kInputRows / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows / 2, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kShardHalf, 1}),
                                        Tensor(out1.p, DType::BF16, {kShardHalf, 1})};
        ops::linear_swiglu_column_parallel(x, {weight, other}, out, ec);
    });
    expect_throw("weight format", [&] {
        Weight other = weight;
        other.qtype  = QType::NVFP4;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kShardHalf, 1}),
                                        Tensor(out1.p, DType::BF16, {kShardHalf, 1})};
        ops::linear_swiglu_column_parallel(x, {weight, other}, out, ec);
    });
    expect_throw("single-device context", [&] {
        const ExecutionContext single({ec.dev[0]->device});
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kShardHalf, 1}),
                                        Tensor(out1.p, DType::BF16, {kShardHalf, 1})};
        ops::linear_swiglu_column_parallel(x, {weight, weight}, out, single);
    });
    std::cout << (failures ? "FAIL" : "OK") << " split rejections\n";
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    try {
        failures = verify_registry();
    } catch (const std::exception& error) {
        std::cerr << "linear_swiglu split registry: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cout << "FAIL linear_swiglu split (registry)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: linear_swiglu split parity requires two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    try {
        const ExecutionContext ec({0, 1});
        std::cout << "peer access: " << (ops::enable_peer_access(ec) ? "direct" : "host-staged")
                  << '\n';
        const ops::PeerEvents events(ec);
        failures += verify_split_rejections(ec);

        constexpr auto kA16 = ops::LinearPolicy::A16Only;
        constexpr auto kA8  = ops::LinearPolicy::AllowA8;
        constexpr auto kA4  = ops::LinearPolicy::AllowA4;
        // Token counts reach every route the half inherits: NVFP4 decode, small-T, fused W4A4
        // (through 128), the materialized linear() + silu_mul() route, and the fused TMA route at
        // whole 256-token tiles; FP8 decode, small-T and A8.
        const std::vector<Case> cases{
            {"nvfp4 gate_up", QType::NVFP4, 31U, {1, 4, 5, 16}, {kA16}},
            {"nvfp4 gate_up", QType::NVFP4, 32U, {1, 4, 5, 16, 128, 129, 300, 1024}, {kA4}},
            {"fp8 gate_up",
             QType::FP8_E4M3FN_ROW_BF16,
             33U,
             {1, 2, 3, 4, 5, 48, 128, 1024},
             {kA16, kA8}},
        };
        for (const Case& test_case : cases) { failures += run_case(test_case, ec); }

        // FP8 A8 is left out: its per-token activation scale covers the whole K row, so the row
        // split changes the quantization; test_linear_add_split covers that route with its own
        // bound.
        const std::vector<MlpCase> mlp_cases{
            {"nvfp4", QType::NVFP4, 41U, 8, kA4},
            {"nvfp4", QType::NVFP4, 42U, 1024, kA4},
            {"fp8", QType::FP8_E4M3FN_ROW_BF16, 43U, 8, kA16},
            {"fp8", QType::FP8_E4M3FN_ROW_BF16, 44U, 1024, kA16},
        };
        for (const MlpCase& test_case : mlp_cases) {
            failures += run_mlp_case(test_case, ec, events);
        }
    } catch (const std::exception& error) {
        std::cerr << "linear_swiglu split: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " linear_swiglu split\n";
    return failures ? 1 : 0;
}
