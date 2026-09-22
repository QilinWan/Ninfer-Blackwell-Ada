// Two-device parity of linear_column_parallel and linear_row_parallel.
//
// The per-format Linear suites qualify every registered problem, the two-device halves included,
// against the FP64 oracle. This suite checks what they cannot: that splitting a projection across
// two devices reproduces the single-device result. Each case runs linear() on device 0 over the
// whole weight and the split form over the two shards, then compares the two.
//
// The whole weight and its shards are generated independently from one logical matrix: the
// quantized fixture keys every code and scale on the global coordinate, so a shard generated with
// an origin is a standalone tensor holding the parent's block, as a two-device loader would
// produce. Every case first checks that through the fixture's independent decoder, and that the
// two shards differ, so a generator defect fails as a weight mismatch rather than passing as a
// comparison of a block with a copy of itself.
//
// Column-parallel ranks evaluate the dot products the whole-weight kernel evaluates for their
// rows, so the result is expected to match to rounding. Row-parallel ranks each round a partial
// to BF16 before the all-reduce adds them, so the bound is stated against the largest output
// rather than per element. The FP8 A8 route also quantizes each rank's activation block with its
// own per-token scale, which is a different quantization from the whole row's; its row-parallel
// cases use the FP8 A8 tolerance.
//
// Every case needs two CUDA devices in one process and the suite reports 77 with fewer. The
// registry probe is host-only and runs first.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "core/weight.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw   = ninfer::test::quantized_weight;
namespace bf16 = ninfer::test::direct_bf16_weight;

namespace {

constexpr double kBf16Ulp = 1.0 / 256.0;

enum class SplitAxis : std::uint8_t {
    Column, // Output rows split; no communication.
    Row,    // Input columns split; all-reduced.
};

struct Case {
    const char* label;
    QType qtype;
    SplitAxis axis;
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    std::vector<std::int32_t> tokens;
    std::vector<ops::LinearPolicy> policies;
};

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
// DeviceContext streams do not order against, so they are retired before a split form runs.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

qw::PackedWeight make_weight(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                             std::int32_t row_origin, std::int32_t column_origin) {
    qw::PatternedWeightOptions options;
    options.row_origin              = row_origin;
    options.column_origin           = column_origin;
    options.decorrelate_coordinates = true;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    }
    return qw::make_patterned_weight(qtype, n, k, seed, options);
}

bf16::HostWeight bf16_block(const bf16::HostWeight& parent, std::int32_t n, std::int32_t k,
                            std::int32_t row_origin, std::int32_t column_origin) {
    bf16::HostWeight block;
    block.n = n;
    block.k = k;
    block.bits.resize(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        const auto source = parent.bits.begin() +
                            static_cast<std::ptrdiff_t>(row + row_origin) * parent.k +
                            column_origin;
        std::copy(source, source + k, block.bits.begin() + static_cast<std::ptrdiff_t>(row) * k);
    }
    return block;
}

// Coordinates straddling the NVFP4 32- and 128-row scale tiles, the quantization groups, and the
// shard's own first and last index.
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

int verify_shard_is_parent_block(const std::string& label, const qw::PackedWeight& parent,
                                 const qw::PackedWeight& shard, std::int32_t row_origin,
                                 std::int32_t column_origin) {
    for (const std::int32_t row : seam_samples(shard.weight.n)) {
        for (const std::int32_t column : seam_samples(shard.weight.k)) {
            const double got = qw::logical_weight_fp64(shard, row, column);
            const double expected =
                qw::logical_weight_fp64(parent, row + row_origin, column + column_origin);
            if (got != expected) {
                std::cerr << label << ": shard (" << row << ',' << column
                          << ") is not the parent's (" << row + row_origin << ','
                          << column + column_origin << "): " << got << " vs " << expected << '\n';
                return 1;
            }
        }
    }
    return 0;
}

template <typename Bytes>
int verify_shards_are_distinct(const std::string& label, const Bytes& first, const Bytes& second) {
    if (first != second) { return 0; }
    std::cerr << label << ": the two shard payloads are byte-identical\n";
    return 1;
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

RankWeight upload(const bf16::HostWeight& host) {
    RankWeight result;
    result.payload = to_device(host.bits);
    result.weight  = host.device_weight(result.payload.p);
    return result;
}

std::size_t workspace_bytes(QType qtype, std::int32_t n, std::int32_t k, ops::LinearPolicy policy,
                            std::int32_t tokens) {
    return std::max<std::size_t>(
        ops::linear_workspace_capacity_bytes(qtype, n, k, policy, tokens, tokens), 1);
}

int compare(const std::string& label, const std::vector<double>& got,
            const std::vector<double>& expected, const ReductionCriterion& criterion) {
    const ReductionStats stats =
        compute_reduction_stats(got.data(), expected.data(), static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2
              << " gross_limit=" << gross_error_limit(stats, criterion) << '\n';
    return verify_reduction(label, got, expected, criterion);
}

// Two BF16 ulp of the largest output, and two ulp of relative L2. A row split differs from the
// whole-K evaluation by two extra BF16 roundings of partials comparable to the result, so about
// one ulp is the expected difference rather than the limit.
constexpr ReductionCriterion kSplitCriterion{2.0 * kBf16Ulp, 0.0, 2.0 * kBf16Ulp};
// The FP8 A8 Linear tolerance: a row split changes each rank's per-token activation scale.
constexpr ReductionCriterion kFp8A8RowSplitCriterion{0.04, kBf16Ulp, 0.06};

ReductionCriterion criterion_for(const Case& test_case, ops::LinearPolicy policy) {
    if (test_case.qtype == QType::FP8_E4M3FN_ROW_BF16 && test_case.axis == SplitAxis::Row &&
        ops::allows_a8(policy)) {
        return kFp8A8RowSplitCriterion;
    }
    return kSplitCriterion;
}

int run_case(const Case& test_case, const ExecutionContext& ec, const ops::PeerEvents& events) {
    const bool column      = test_case.axis == SplitAxis::Column;
    const bool dense       = test_case.qtype == QType::BF16;
    const std::int32_t n   = test_case.n;
    const std::int32_t k   = test_case.k;
    const std::int32_t sn  = column ? n / 2 : n;
    const std::int32_t sk  = column ? k : k / 2;
    const std::string head = std::string(test_case.label) + (column ? " column" : " row");
    std::cout << head << " [" << n << ',' << k << "] -> [" << sn << ',' << sk << "]\n";

    int failures = 0;
    std::optional<qw::PackedWeight> packed_parent;
    std::optional<bf16::HostWeight> dense_parent;
    std::array<std::optional<qw::PackedWeight>, 2> packed_shard;
    std::array<std::optional<bf16::HostWeight>, 2> dense_shard;
    if (dense) {
        dense_parent.emplace(bf16::make_patterned(n, k, test_case.seed));
    } else {
        packed_parent.emplace(make_weight(test_case.qtype, n, k, test_case.seed, 0, 0));
    }
    for (std::size_t rank = 0; rank < 2; ++rank) {
        const std::int32_t row_origin    = column ? static_cast<std::int32_t>(rank) * sn : 0;
        const std::int32_t column_origin = column ? 0 : static_cast<std::int32_t>(rank) * sk;
        if (dense) {
            dense_shard[rank].emplace(bf16_block(*dense_parent, sn, sk, row_origin, column_origin));
        } else {
            packed_shard[rank].emplace(
                make_weight(test_case.qtype, sn, sk, test_case.seed, row_origin, column_origin));
            failures += verify_shard_is_parent_block(head + " shard " + std::to_string(rank),
                                                     *packed_parent, *packed_shard[rank],
                                                     row_origin, column_origin);
        }
    }
    failures += dense ? verify_shards_are_distinct(head, dense_shard[0]->bits, dense_shard[1]->bits)
                      : verify_shards_are_distinct(head, packed_shard[0]->payload,
                                                   packed_shard[1]->payload);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    const RankWeight parent = dense ? upload(*dense_parent) : upload(*packed_parent);
    std::array<RankWeight, 2> shard;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        shard[rank] = dense ? upload(*dense_shard[rank]) : upload(*packed_shard[rank]);
    }

    for (const std::int32_t tokens : test_case.tokens) {
        std::vector<float> activation(static_cast<std::size_t>(k) * tokens);
        fill_uniform(activation, test_case.seed * 31U + static_cast<std::uint32_t>(tokens), -1.0F,
                     1.0F);
        round_to_bf16(activation);
        set_device(ec, 0);
        const DeviceBuffer parent_x = to_device_bf16(activation);
        std::array<DeviceBuffer, 2> shard_x;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            std::vector<float> block(static_cast<std::size_t>(sk) * tokens);
            for (std::int32_t token = 0; token < tokens; ++token) {
                const std::size_t source = static_cast<std::size_t>(token) * k +
                                           (column ? 0 : rank * static_cast<std::size_t>(sk));
                std::copy(activation.begin() + static_cast<std::ptrdiff_t>(source),
                          activation.begin() + static_cast<std::ptrdiff_t>(source + sk),
                          block.begin() + static_cast<std::ptrdiff_t>(token) * sk);
            }
            set_device(ec, static_cast<int>(rank));
            shard_x[rank] = to_device_bf16(block);
        }

        for (const ops::LinearPolicy policy : test_case.policies) {
            const std::string label =
                head + " T=" + std::to_string(tokens) + " " + policy_name(policy);

            const std::size_t parent_elements = static_cast<std::size_t>(n) * tokens;
            set_device(ec, 0);
            GuardedDeviceBuffer reference(parent_elements * sizeof(std::uint16_t));
            reference.fill(0xff);
            DeviceArena reference_arena(workspace_bytes(test_case.qtype, n, k, policy, tokens));
            const Tensor reference_x(parent_x.p, DType::BF16, {k, tokens});
            Tensor reference_out(reference.data(), DType::BF16, {n, tokens});
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear(reference_x, parent.weight, reference_out, policy, reference_arena,
                        ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += reference.verify_guards(label + " reference");
            const std::vector<double> expected =
                from_device_bf16(reference.data(), parent_elements);

            const std::size_t shard_elements = static_cast<std::size_t>(sn) * tokens;
            std::array<std::optional<GuardedDeviceBuffer>, 2> shard_out;
            std::array<std::optional<DeviceBuffer>, 2> staging;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                set_device(ec, static_cast<int>(rank));
                shard_out[rank].emplace(shard_elements * sizeof(std::uint16_t));
                shard_out[rank]->fill(0xff);
                staging[rank].emplace(shard_elements * sizeof(std::uint16_t));
                arena[rank].emplace(workspace_bytes(test_case.qtype, sn, sk, policy, tokens));
            }
            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {sk, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {sk, tokens})};
            const std::array<Weight, 2> w{shard[0].weight, shard[1].weight};
            const std::array<Tensor, 2> out{
                Tensor(shard_out[0]->data(), DType::BF16, {sn, tokens}),
                Tensor(shard_out[1]->data(), DType::BF16, {sn, tokens})};
            const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            if (column) {
                ops::linear_column_parallel(x, w, out, policy, workspace, ec);
            } else {
                const std::array<Tensor, 2> staging_view{
                    Tensor(staging[0]->p, DType::BF16, {sn, tokens}),
                    Tensor(staging[1]->p, DType::BF16, {sn, tokens})};
                ops::linear_row_parallel(x, w, out, staging_view, policy, workspace, ec, events);
            }
            synchronize_both(ec);

            const ReductionCriterion criterion = criterion_for(test_case, policy);
            std::array<std::vector<double>, 2> observed;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                const std::string rank_label = label + " rank " + std::to_string(rank);
                set_device(ec, static_cast<int>(rank));
                failures += shard_out[rank]->verify_guards(rank_label);
                observed[rank] = from_device_bf16(shard_out[rank]->data(), shard_elements);
                if (!column) {
                    failures += compare(rank_label, observed[rank], expected, criterion);
                    continue;
                }
                // Rank r owns output rows [r*sn, (r+1)*sn) of every token.
                std::vector<double> block(shard_elements);
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const std::size_t source =
                        static_cast<std::size_t>(token) * n + rank * static_cast<std::size_t>(sn);
                    std::copy(expected.begin() + static_cast<std::ptrdiff_t>(source),
                              expected.begin() + static_cast<std::ptrdiff_t>(source + sn),
                              block.begin() + static_cast<std::ptrdiff_t>(token) * sn);
                }
                failures += compare(rank_label, observed[rank], block, criterion);
            }
            if (column && observed[0] == observed[1]) {
                std::cerr << label << ": both ranks produced the same output block\n";
                ++failures;
            }
            // The all-reduce combines the same operands on both ranks, so the sums are identical.
            if (!column && observed[0] != observed[1]) {
                std::cerr << label << ": the ranks disagree after the all-reduce\n";
                ++failures;
            }
        }
    }
    return failures;
}

// linear_workspace_capacity_bytes() runs each format's shape resolver without a device, so the
// registry of two-device halves is checked even where the parity cases must skip.
int verify_registry() {
    struct Entry {
        QType qtype;
        std::int32_t n;
        std::int32_t k;
    };

    const std::vector<Entry> admitted{
        {QType::FP8_E4M3FN_ROW_BF16, 7168, 5120},
        {QType::FP8_E4M3FN_ROW_BF16, 8192, 5120},
        {QType::FP8_E4M3FN_ROW_BF16, 17408, 5120},
        {QType::FP8_E4M3FN_ROW_BF16, 124160, 5120},
        {QType::FP8_E4M3FN_ROW_BF16, 5120, 3072},
        {QType::FP8_E4M3FN_ROW_BF16, 5120, 8704},
        {QType::NVFP4, 17408, 5120},
        {QType::NVFP4, 5120, 8704},
        {QType::BF16, 7168, 5120},
        {QType::BF16, 5120, 3072},
    };
    constexpr std::array policies{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                  ops::LinearPolicy::AllowA4};
    int failures = 0;
    for (const Entry& entry : admitted) {
        for (const ops::LinearPolicy policy : policies) {
            for (const std::int32_t tokens : {1, 2, 48, 1024}) {
                try {
                    (void)ops::linear_workspace_capacity_bytes(entry.qtype, entry.n, entry.k,
                                                               policy, tokens, tokens);
                } catch (const std::exception& error) {
                    std::cerr << "registry: [" << entry.n << ',' << entry.k << "] qtype "
                              << static_cast<int>(entry.qtype) << ' ' << policy_name(policy)
                              << " T=" << tokens << " rejected: " << error.what() << '\n';
                    ++failures;
                }
            }
        }
    }
    const std::vector<Entry> rejected{
        {QType::FP8_E4M3FN_ROW_BF16, 5120, 5120},
        {QType::NVFP4, 5120, 5120},
        {QType::BF16, 8192, 5120},
    };
    for (const Entry& entry : rejected) {
        try {
            (void)ops::linear_workspace_capacity_bytes(entry.qtype, entry.n, entry.k,
                                                       ops::LinearPolicy::A16Only, 1, 1);
            std::cerr << "registry: [" << entry.n << ',' << entry.k << "] qtype "
                      << static_cast<int>(entry.qtype) << " was admitted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    std::cout << (failures ? "FAIL" : "OK") << " registry: " << admitted.size()
              << " two-device halves admitted\n";
    return failures;
}

int verify_split_rejections(const ExecutionContext& ec, const ops::PeerEvents& events) {
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

    constexpr std::int32_t kN = 7168;
    constexpr std::int32_t kK = 5120;
    const std::size_t x_bytes = static_cast<std::size_t>(kK) * 2 * sizeof(std::uint16_t);
    const std::size_t y_bytes = static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t);
    set_device(ec, 0);
    DeviceBuffer x0(x_bytes);
    DeviceBuffer out0(y_bytes);
    DeviceBuffer stage0(y_bytes);
    set_device(ec, 1);
    DeviceBuffer x1(x_bytes);
    DeviceBuffer out1(y_bytes);
    DeviceBuffer stage1(y_bytes);

    Weight weight{};
    weight.qtype = QType::FP8_E4M3FN_ROW_BF16;
    weight.n     = kN;
    weight.k     = kK;

    expect_throw("token count", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 2}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kN, 2}),
                                        Tensor(out1.p, DType::BF16, {kN, 1})};
        ops::linear_column_parallel(x, {weight, weight}, out, ec);
    });
    expect_throw("column K", [&] {
        Weight other = weight;
        other.k      = kK / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK / 2, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kN, 1}),
                                        Tensor(out1.p, DType::BF16, {kN, 1})};
        ops::linear_column_parallel(x, {weight, other}, out, ec);
    });
    expect_throw("row N", [&] {
        Weight other = weight;
        other.n      = kN / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kN, 1}),
                                        Tensor(out1.p, DType::BF16, {kN / 2, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 1}),
                                            Tensor(stage1.p, DType::BF16, {kN / 2, 1})};
        ops::linear_row_parallel(x, {weight, other}, out, staging, ec, events);
    });
    expect_throw("single-device context", [&] {
        const ExecutionContext single({ec.dev[0]->device});
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kN, 1}),
                                        Tensor(out1.p, DType::BF16, {kN, 1})};
        ops::linear_column_parallel(x, {weight, weight}, out, single);
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
        std::cerr << "linear split registry: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cout << "FAIL linear split (registry)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: linear split parity requires two CUDA devices, found " << device_count
                  << '\n';
        return 77;
    }

    try {
        const ExecutionContext ec({0, 1});
        std::cout << "peer access: " << (ops::enable_peer_access(ec) ? "direct" : "host-staged")
                  << '\n';
        const ops::PeerEvents events(ec);
        failures += verify_split_rejections(ec, events);

        constexpr auto kA16 = ops::LinearPolicy::A16Only;
        constexpr auto kA8  = ops::LinearPolicy::AllowA8;
        constexpr auto kA4  = ops::LinearPolicy::AllowA4;
        // Token counts reach each half's decode, SIMT-chunk, A8/A4 crossover, MMA, and (at 1024)
        // TMA routes, which the halves inherit from the problem they split.
        const std::vector<Case> cases{
            {"fp8 attention input",
             QType::FP8_E4M3FN_ROW_BF16,
             SplitAxis::Column,
             14336,
             5120,
             11U,
             {1, 8, 11, 12, 48, 1024},
             {kA16, kA8}},
            {"fp8 gdn input",
             QType::FP8_E4M3FN_ROW_BF16,
             SplitAxis::Column,
             16384,
             5120,
             12U,
             {1, 10, 11, 128, 1024},
             {kA16, kA8}},
            {"fp8 mlp gate_up",
             QType::FP8_E4M3FN_ROW_BF16,
             SplitAxis::Column,
             34816,
             5120,
             13U,
             {1, 4, 5, 48, 1024},
             {kA16, kA8}},
            {"nvfp4 mlp gate_up",
             QType::NVFP4,
             SplitAxis::Column,
             34816,
             5120,
             14U,
             {1, 5, 32, 48, 128, 256, 1024},
             {kA16, kA4}},
            {"bf16 attention input",
             QType::BF16,
             SplitAxis::Column,
             14336,
             5120,
             15U,
             {1, 8, 48},
             {kA16}},
            {"fp8 output",
             QType::FP8_E4M3FN_ROW_BF16,
             SplitAxis::Row,
             5120,
             6144,
             21U,
             {1, 8, 24, 25, 48, 1024},
             {kA16, kA8}},
            {"fp8 mlp down",
             QType::FP8_E4M3FN_ROW_BF16,
             SplitAxis::Row,
             5120,
             17408,
             22U,
             {1, 8, 24, 25, 48, 1024},
             {kA16, kA8}},
            {"nvfp4 mlp down",
             QType::NVFP4,
             SplitAxis::Row,
             5120,
             17408,
             23U,
             {1, 8, 48, 128, 512, 1024},
             {kA16, kA4}},
            {"bf16 output", QType::BF16, SplitAxis::Row, 5120, 6144, 24U, {1, 8, 48}, {kA16}},
        };
        for (const Case& test_case : cases) { failures += run_case(test_case, ec, events); }
    } catch (const std::exception& error) {
        std::cerr << "linear split: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " linear split\n";
    return failures ? 1 : 0;
}
