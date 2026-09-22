// Two-device parity of attn_input_proj_column_parallel.
//
// The single-parent projection splits by heads: rank r's standalone [7168,5120] shard stacks rank
// r's halves of the parent's query [0,6144), key [6144,7168), gate [7168,13312) and value
// [13312,14336) sections in that order, so its sections sit at shard rows 0 (query, 3072),
// 3072 (key, 512), 3584 (gate, 3072) and 6656 (value, 512). Each case runs attn_input_proj() on
// device 0 over the whole [14336,5120] parent and the split form over the two shards, then
// compares every rank's four outputs with the matching head blocks of the single-device outputs.
//
// The parent and the eight section blocks are generated independently from one logical matrix
// (the quantized fixture keys codes and scales on the global coordinate) and each shard is the
// splice of its four blocks. Every case first checks through the fixture's decoder that each block
// is the parent's block at its origin and that the ranks' shards differ, so a fixture or splice
// defect fails as a weight mismatch instead of passing as a comparison with itself.
//
// Each rank evaluates, per output row, the dot product the single-device kernel evaluates for the
// same parent row, so two BF16 ulp of the largest output bounds the difference. Only FP8 shards
// are registered. Every case needs two CUDA devices in one process and the suite reports 77 with
// fewer. The registry probe is host-only and runs first.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/attn_input_proj.h"

#include "core/device.h"
#include "core/weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

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

constexpr double kBf16Ulp = 1.0 / 256.0;

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kQRows      = 6144;
constexpr std::int32_t kKvRows     = 1024;
constexpr std::int32_t kParentRows = 2 * (kQRows + kKvRows);
constexpr std::int32_t kShardQRows = kQRows / 2;
constexpr std::int32_t kShardKv    = kKvRows / 2;
constexpr std::int32_t kShardRows  = 2 * (kShardQRows + kShardKv);

// Parent row where each section begins, in the parent's query, key, gate, value order.
constexpr std::array<std::int32_t, 4> kSectionBegin{0, kQRows, kQRows + kKvRows,
                                                    2 * kQRows + kKvRows};
constexpr std::array<std::int32_t, 4> kShardSectionRows{kShardQRows, kShardKv, kShardQRows,
                                                        kShardKv};
constexpr std::array<const char*, 4> kSectionName{"q", "k", "gate", "v"};

// Two BF16 ulp of the largest output, and two ulp of relative L2.
constexpr ReductionCriterion kSplitCriterion{2.0 * kBf16Ulp, 0.0, 2.0 * kBf16Ulp};

const char* policy_name(ops::LinearPolicy policy) {
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

void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

void synchronize_both(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }
}

// Uploads and poison fills run on each device's legacy default stream, which the non-blocking
// DeviceContext streams do not order against, so they are retired before the split form runs.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

qw::PackedWeight make_fp8(std::int32_t n, std::uint32_t seed, std::int32_t row_origin) {
    qw::PatternedWeightOptions options;
    options.row_origin              = row_origin;
    options.decorrelate_coordinates = true;
    return qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, n, kHidden, seed, options);
}

// Stacks FP8 row blocks of one K into one standalone weight: the codes, then the row scales at the
// next 256-byte boundary, the layout the fixture and validate_fp8_weight() use.
qw::PackedWeight stack_fp8_rows(const std::vector<qw::PackedWeight>& blocks) {
    std::int32_t n = 0;
    for (const qw::PackedWeight& block : blocks) {
        if (block.weight.qtype != QType::FP8_E4M3FN_ROW_BF16 || block.weight.k != kHidden) {
            throw std::invalid_argument("stack_fp8_rows: blocks must be FP8 with K=5120");
        }
        n += block.weight.n;
    }
    qw::PackedWeight stacked = blocks.front();
    stacked.code_plane_bytes = static_cast<std::uint64_t>(n) * kHidden;
    stacked.scale_plane_offset =
        (stacked.code_plane_bytes + 255U) & ~static_cast<std::uint64_t>(255U);
    stacked.scale_plane_bytes = static_cast<std::uint64_t>(n) * 2;
    stacked.payload.assign(stacked.scale_plane_offset + stacked.scale_plane_bytes, 0);
    std::size_t code_offset  = 0;
    std::size_t scale_offset = stacked.scale_plane_offset;
    for (const qw::PackedWeight& block : blocks) {
        std::memcpy(stacked.payload.data() + code_offset, block.payload.data(),
                    block.code_plane_bytes);
        std::memcpy(stacked.payload.data() + scale_offset,
                    block.payload.data() + block.scale_plane_offset, block.scale_plane_bytes);
        code_offset += block.code_plane_bytes;
        scale_offset += block.scale_plane_bytes;
    }
    Weight& weight         = stacked.weight;
    weight.payload         = stacked.payload.data();
    weight.payload_bytes   = stacked.payload.size();
    weight.qdata           = stacked.payload.data();
    weight.scales          = stacked.payload.data() + stacked.scale_plane_offset;
    weight.n               = n;
    weight.shape[0]        = n;
    weight.padded_shape[0] = n;
    weight.scale_ne[0]     = n;
    weight.scale_nb[1]     = static_cast<std::int64_t>(n) * 2;
    weight.scale_nb[2]     = weight.scale_nb[1];
    weight.scale_nb[3]     = weight.scale_nb[1];
    stacked.dequant.clear();
    return stacked;
}

// Rows and columns straddling the FP8 tiles, the section boundaries, and the block's last index.
std::vector<std::int32_t> seam_samples(std::int32_t extent) {
    const std::vector<std::int32_t> probes{0,  1,   15,  16,  31,  32,         63,
                                           64, 127, 128, 511, 512, extent / 2, extent - 1};
    std::vector<std::int32_t> result;
    for (const std::int32_t probe : probes) {
        if (probe >= 0 && probe < extent &&
            std::find(result.begin(), result.end(), probe) == result.end()) {
            result.push_back(probe);
        }
    }
    return result;
}

// Checks that `shard` rows [shard_row, shard_row + rows) hold the parent's rows from `parent_row`.
int verify_section(const std::string& label, const qw::PackedWeight& parent,
                   const qw::PackedWeight& shard, std::int32_t parent_row, std::int32_t shard_row,
                   std::int32_t rows) {
    for (const std::int32_t row : seam_samples(rows)) {
        for (const std::int32_t column : seam_samples(kHidden)) {
            const double got      = qw::logical_weight_fp64(shard, shard_row + row, column);
            const double expected = qw::logical_weight_fp64(parent, parent_row + row, column);
            if (got != expected) {
                std::cerr << label << ": shard row " << shard_row + row << " is not parent row "
                          << parent_row + row << " at column " << column << ": " << got << " vs "
                          << expected << '\n';
                return 1;
            }
        }
    }
    return 0;
}

struct RankWeight {
    DeviceBuffer payload;
    Weight weight{};
};

RankWeight upload(const qw::PackedWeight& packed) {
    RankWeight result;
    result.payload = to_device(packed.payload);
    result.weight  = packed.device_weight(result.payload.p);
    return result;
}

int compare(const std::string& label, const std::vector<double>& got,
            const std::vector<double>& expected) {
    const ReductionStats stats =
        compute_reduction_stats(got.data(), expected.data(), static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2
              << " gross_limit=" << gross_error_limit(stats, kSplitCriterion)
              << (got == expected ? " (bit-identical)" : "") << '\n';
    return verify_reduction(label, got, expected, kSplitCriterion);
}

// Rows [begin, begin + rows) of every token of a [parent_rows, tokens] output.
std::vector<double> row_block(const std::vector<double>& output, std::int32_t parent_rows,
                              std::int32_t begin, std::int32_t rows, std::int32_t tokens) {
    std::vector<double> block(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const auto source =
            output.begin() + static_cast<std::ptrdiff_t>(token) * parent_rows + begin;
        std::copy(source, source + rows, block.begin() + static_cast<std::ptrdiff_t>(token) * rows);
    }
    return block;
}

std::size_t bf16_bytes(std::int32_t rows, std::int32_t tokens) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(tokens) *
           sizeof(std::uint16_t);
}

int run_fp8_case(const ExecutionContext& ec, std::uint32_t seed,
                 const std::vector<std::int32_t>& token_counts,
                 const std::vector<ops::LinearPolicy>& policies) {
    const std::string head = "fp8 attn_input_proj";
    std::cout << head << " [" << kParentRows << ',' << kHidden << "] -> [" << kShardRows << ','
              << kHidden << "] per rank\n";

    int failures                  = 0;
    const qw::PackedWeight parent = make_fp8(kParentRows, seed, 0);
    std::array<std::optional<qw::PackedWeight>, 2> shard;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<qw::PackedWeight> blocks;
        for (std::size_t section = 0; section < 4; ++section) {
            const std::int32_t rows       = kShardSectionRows[section];
            const std::int32_t parent_row = kSectionBegin[section] + rank * rows;
            blocks.push_back(make_fp8(rows, seed, parent_row));
            failures += verify_section(head + " rank " + std::to_string(rank) + " " +
                                           kSectionName[section] + " block",
                                       parent, blocks.back(), parent_row, 0, rows);
        }
        shard[static_cast<std::size_t>(rank)].emplace(stack_fp8_rows(blocks));
        std::int32_t shard_row = 0;
        for (std::size_t section = 0; section < 4; ++section) {
            const std::int32_t rows = kShardSectionRows[section];
            failures += verify_section(head + " rank " + std::to_string(rank) + " " +
                                           kSectionName[section] + " in shard",
                                       parent, *shard[static_cast<std::size_t>(rank)],
                                       kSectionBegin[section] + rank * rows, shard_row, rows);
            shard_row += rows;
        }
    }
    if (shard[0]->payload == shard[1]->payload) {
        std::cerr << head << ": the two shard payloads are byte-identical\n";
        ++failures;
    }
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    const RankWeight parent_device = upload(parent);
    std::array<RankWeight, 2> shard_device;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        shard_device[rank] = upload(*shard[rank]);
    }

    for (const std::int32_t tokens : token_counts) {
        std::vector<float> activation(static_cast<std::size_t>(kHidden) * tokens);
        fill_uniform(activation, seed * 31U + static_cast<std::uint32_t>(tokens), -1.0F, 1.0F);
        round_to_bf16(activation);
        set_device(ec, 0);
        const DeviceBuffer parent_x = to_device_bf16(activation);
        std::array<DeviceBuffer, 2> shard_x;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            set_device(ec, static_cast<int>(rank));
            shard_x[rank] = to_device_bf16(activation);
        }

        for (const ops::LinearPolicy policy : policies) {
            const std::string label =
                head + " T=" + std::to_string(tokens) + " " + policy_name(policy);

            // Single-device reference over the whole parent on device 0.
            set_device(ec, 0);
            constexpr std::array<std::int32_t, 4> kParentSectionRows{kQRows, kKvRows, kQRows,
                                                                     kKvRows};
            std::array<std::optional<GuardedDeviceBuffer>, 4> reference;
            for (std::size_t section = 0; section < 4; ++section) {
                reference[section].emplace(bf16_bytes(kParentSectionRows[section], tokens));
                reference[section]->fill(0xff);
            }
            const std::size_t reference_capacity = ops::attn_input_proj_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16, kParentRows, kHidden, policy, tokens, tokens);
            DeviceArena reference_arena(std::max<std::size_t>(reference_capacity, 1));
            const Tensor reference_x(parent_x.p, DType::BF16, {kHidden, tokens});
            Tensor reference_q(reference[0]->data(), DType::BF16, {kQRows, tokens});
            Tensor reference_k(reference[1]->data(), DType::BF16, {kKvRows, tokens});
            Tensor reference_gate(reference[2]->data(), DType::BF16, {kQRows, tokens});
            Tensor reference_v(reference[3]->data(), DType::BF16, {kKvRows, tokens});
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::attn_input_proj(reference_x, parent_device.weight, reference_q, reference_gate,
                                 reference_k, reference_v, policy, reference_arena,
                                 ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            std::array<std::vector<double>, 4> expected;
            for (std::size_t section = 0; section < 4; ++section) {
                failures += reference[section]->verify_guards(label + " reference " +
                                                              kSectionName[section]);
                expected[section] = from_device_bf16(
                    reference[section]->data(),
                    static_cast<std::size_t>(kParentSectionRows[section]) * tokens);
            }

            // Split form: rank r owns heads r of every section.
            const std::size_t split_capacity =
                ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                    QType::FP8_E4M3FN_ROW_BF16, policy, tokens, tokens);
            std::array<std::array<std::optional<GuardedDeviceBuffer>, 4>, 2> output;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                set_device(ec, static_cast<int>(rank));
                for (std::size_t section = 0; section < 4; ++section) {
                    output[rank][section].emplace(bf16_bytes(kShardSectionRows[section], tokens));
                    output[rank][section]->fill(0xff);
                }
                arena[rank].emplace(std::max<std::size_t>(split_capacity, 1));
            }
            const auto section_tensors = [&](std::size_t section) {
                return std::array<Tensor, 2>{Tensor(output[0][section]->data(), DType::BF16,
                                                    {kShardSectionRows[section], tokens}),
                                             Tensor(output[1][section]->data(), DType::BF16,
                                                    {kShardSectionRows[section], tokens})};
            };
            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kHidden, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {kHidden, tokens})};
            const std::array<Weight, 2> w{shard_device[0].weight, shard_device[1].weight};
            const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::attn_input_proj_column_parallel(x, w, section_tensors(0), section_tensors(2),
                                                 section_tensors(1), section_tensors(3), policy,
                                                 workspace, ec);
            synchronize_both(ec);

            std::array<std::vector<double>, 2> observed_q;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                set_device(ec, static_cast<int>(rank));
                for (std::size_t section = 0; section < 4; ++section) {
                    const std::int32_t rows = kShardSectionRows[section];
                    const std::string rank_label =
                        label + " rank " + std::to_string(rank) + " " + kSectionName[section];
                    failures += output[rank][section]->verify_guards(rank_label);
                    const std::vector<double> observed = from_device_bf16(
                        output[rank][section]->data(), static_cast<std::size_t>(rows) * tokens);
                    failures +=
                        compare(rank_label, observed,
                                row_block(expected[section], kParentSectionRows[section],
                                          static_cast<std::int32_t>(rank) * rows, rows, tokens));
                    if (section == 0) { observed_q[rank] = observed; }
                }
            }
            if (observed_q[0] == observed_q[1]) {
                std::cerr << label << ": both ranks produced the same query block\n";
                ++failures;
            }
        }
    }
    return failures;
}

// The workspace query is host-only, so the registry is checked even where parity must skip.
int verify_registry() {
    int failures = 0;
    for (const ops::LinearPolicy policy :
         {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
        for (const std::int32_t tokens : {1, 4, 5, 33, 1024}) {
            try {
                const std::size_t shard =
                    ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                        QType::FP8_E4M3FN_ROW_BF16, policy, tokens, tokens);
                const std::size_t whole = ops::attn_input_proj_workspace_capacity_bytes(
                    QType::FP8_E4M3FN_ROW_BF16, kParentRows, kHidden, policy, tokens, tokens);
                if (shard != whole) {
                    std::cerr << "registry: FP8 shard capacity " << shard
                              << " differs from the parent's " << whole << " at "
                              << policy_name(policy) << " T=" << tokens << '\n';
                    ++failures;
                }
            } catch (const std::exception& error) {
                std::cerr << "registry: FP8 shard " << policy_name(policy) << " T=" << tokens
                          << " rejected: " << error.what() << '\n';
                ++failures;
            }
        }
    }
    for (const QType qtype : {QType::BF16, QType::NVFP4, QType::Q4_G64_FP16, QType::Q8_G32_FP16}) {
        try {
            (void)ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                qtype, ops::LinearPolicy::A16Only, 1, 1);
            std::cerr << "registry: qtype " << static_cast<int>(qtype) << " was admitted\n";
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

    std::array<std::array<std::optional<DeviceBuffer>, 5>, 2> buffers;
    constexpr std::array<std::int32_t, 5> kRows{kHidden, kShardQRows, kShardQRows, kShardKv,
                                                kShardKv};
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        for (std::size_t slot = 0; slot < 5; ++slot) {
            buffers[rank][slot].emplace(bf16_bytes(kRows[slot], 2));
        }
    }
    // x, q, gate, k, v with tokens[r] columns on rank r.
    const auto tensors = [&](std::size_t slot, std::array<std::int32_t, 2> tokens,
                             std::array<std::int32_t, 2> rows) {
        return std::array<Tensor, 2>{
            Tensor(buffers[0][slot]->p, DType::BF16, {rows[0], tokens[0]}),
            Tensor(buffers[1][slot]->p, DType::BF16, {rows[1], tokens[1]})};
    };
    const auto call = [&](std::array<std::int32_t, 2> tokens, std::array<Weight, 2> w,
                          std::array<std::int32_t, 2> hidden, const ExecutionContext& context) {
        ops::attn_input_proj_column_parallel(tensors(0, tokens, hidden), w,
                                             tensors(1, tokens, {kShardQRows, kShardQRows}),
                                             tensors(2, tokens, {kShardQRows, kShardQRows}),
                                             tensors(3, tokens, {kShardKv, kShardKv}),
                                             tensors(4, tokens, {kShardKv, kShardKv}), context);
    };

    Weight weight{};
    weight.qtype  = QType::FP8_E4M3FN_ROW_BF16;
    weight.layout = QuantLayout::RowScale;
    weight.n      = kShardRows;
    weight.k      = kHidden;

    expect_throw("token count", [&] { call({2, 1}, {weight, weight}, {kHidden, kHidden}, ec); });
    expect_throw("K", [&] {
        Weight other = weight;
        other.k      = kHidden / 2;
        call({1, 1}, {weight, other}, {kHidden, kHidden / 2}, ec);
    });
    expect_throw("weight format", [&] {
        Weight other = weight;
        other.qtype  = QType::NVFP4;
        other.layout = QuantLayout::BlockScaleK16M128x4;
        call({1, 1}, {weight, other}, {kHidden, kHidden}, ec);
    });
    expect_throw("unregistered NVFP4 shard", [&] {
        Weight nvfp4 = weight;
        nvfp4.qtype  = QType::NVFP4;
        nvfp4.layout = QuantLayout::BlockScaleK16M128x4;
        call({1, 1}, {nvfp4, nvfp4}, {kHidden, kHidden}, ec);
    });
    expect_throw("single-device context", [&] {
        const ExecutionContext single({ec.dev[0]->device});
        call({1, 1}, {weight, weight}, {kHidden, kHidden}, single);
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
        std::cerr << "attn_input_proj split registry: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cout << "FAIL attn_input_proj split (registry)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: attn_input_proj split parity requires two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    try {
        const ExecutionContext ec({0, 1});
        std::cout << "peer access: " << (ops::enable_peer_access(ec) ? "direct" : "host-staged")
                  << '\n';
        failures += verify_split_rejections(ec);
        // T reaches the A16 decode (1), SIMT (2..5), K-split MMA (6..33) and GEMM routes, the A8
        // crossover (5) and each A8 tile band up to the prefill tile.
        failures += run_fp8_case(
            ec, 46U, {1, 2, 4, 5, 6, 33, 34, 64, 65, 129, 145, 1024},
            {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4});
    } catch (const std::exception& error) {
        std::cerr << "attn_input_proj split: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " attn_input_proj split\n";
    return failures ? 1 : 0;
}
