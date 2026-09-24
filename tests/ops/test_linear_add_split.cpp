// Two-device parity of linear_add_row_parallel.
//
// The per-format LinearAdd suites qualify the registered halves against the FP64 oracle. This
// suite checks that splitting the fused-residual projection across two devices reproduces the
// single-device result: each case runs linear_add() on device 0 over the whole weight and the
// row-parallel form over the two input-column shards, both starting from the same residual, then
// compares the two. The residual must enter the sum exactly once; adding it on both ranks or on
// neither moves the result by the residual's own magnitude, far outside the bound, so the parity
// comparison covers that as well.
//
// The whole weight and its shards are generated independently from one logical matrix, as in
// test_linear_split, and every case first checks through the fixture's independent decoder that
// each shard is its parent's block and that the two shards differ.
//
// Each rank rounds its partial to BF16 before the all-reduce adds them, so the bound is stated
// against the largest output rather than per element. The FP8 A8 route also quantizes each rank's
// activation block with its own per-token scale, which is a different quantization from the whole
// row's; its cases use the FP8 A8 tolerance.
//
// Every case needs two CUDA devices in one process and the suite reports 77 with fewer. The
// registry probe is host-only and runs first.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear_add.h"

#include "core/device.h"
#include "core/weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/split_test_support.h"

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
namespace qw = ninfer::test::quantized_weight;

namespace {

struct Case {
    const char* label;
    QType qtype;
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    std::vector<std::int32_t> tokens;
    std::vector<ops::LinearPolicy> policies;
};

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
                                 const qw::PackedWeight& shard, std::int32_t column_origin) {
    for (const std::int32_t row : seam_samples(shard.weight.n)) {
        for (const std::int32_t column : seam_samples(shard.weight.k)) {
            const double got      = qw::logical_weight_fp64(shard, row, column);
            const double expected = qw::logical_weight_fp64(parent, row, column + column_origin);
            if (got != expected) {
                std::cerr << label << ": shard (" << row << ',' << column
                          << ") is not the parent's (" << row << ',' << column + column_origin
                          << "): " << got << " vs " << expected << '\n';
                return 1;
            }
        }
    }
    return 0;
}

// 2u of the largest output (one to two BF16 ulp, see kBf16UnitRoundoff) and of relative L2: the
// split adds two BF16 roundings of partials comparable to the result, so about one ulp is the
// expected difference.
constexpr ReductionCriterion kSplitCriterion{2.0 * kBf16UnitRoundoff, 0.0, 2.0 * kBf16UnitRoundoff};
// The FP8 A8 Linear tolerance: a row split changes each rank's per-token activation scale.
constexpr ReductionCriterion kFp8A8RowSplitCriterion{0.04, kBf16UnitRoundoff, 0.06};

ReductionCriterion criterion_for(const Case& test_case, ops::LinearPolicy policy) {
    if (test_case.qtype == QType::FP8_E4M3FN_ROW_BF16 && ops::allows_a8(policy)) {
        return kFp8A8RowSplitCriterion;
    }
    return kSplitCriterion;
}

int run_case(const Case& test_case, const ExecutionContext& ec, const ops::PeerEvents& events) {
    const std::int32_t n   = test_case.n;
    const std::int32_t k   = test_case.k;
    const std::int32_t sk  = k / 2;
    const std::string head = test_case.label;
    std::cout << head << " [" << n << ',' << k << "] -> [" << n << ',' << sk << "]\n";

    int failures                  = 0;
    const qw::PackedWeight parent = make_weight(test_case.qtype, n, k, test_case.seed, 0, 0);
    std::array<std::optional<qw::PackedWeight>, 2> packed_shard;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        const std::int32_t column_origin = static_cast<std::int32_t>(rank) * sk;
        packed_shard[rank].emplace(
            make_weight(test_case.qtype, n, sk, test_case.seed, 0, column_origin));
        failures += verify_shard_is_parent_block(head + " shard " + std::to_string(rank), parent,
                                                 *packed_shard[rank], column_origin);
    }
    if (packed_shard[0]->payload == packed_shard[1]->payload) {
        std::cerr << head << ": the two shard payloads are byte-identical\n";
        ++failures;
    }
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    const RankWeight parent_device = upload(parent);
    std::array<RankWeight, 2> shard;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        shard[rank] = upload(*packed_shard[rank]);
    }

    for (const std::int32_t tokens : test_case.tokens) {
        std::vector<float> activation(static_cast<std::size_t>(k) * tokens);
        fill_uniform(activation, test_case.seed * 31U + static_cast<std::uint32_t>(tokens), -1.0F,
                     1.0F);
        round_to_bf16(activation);
        std::vector<float> residual_values(static_cast<std::size_t>(n) * tokens);
        fill_uniform(residual_values, test_case.seed * 97U + static_cast<std::uint32_t>(tokens),
                     -2.0F, 2.0F);
        std::vector<std::uint16_t> residual_bits(residual_values.size());
        std::transform(residual_values.begin(), residual_values.end(), residual_bits.begin(),
                       f32_to_bf16);
        const std::size_t elements = residual_bits.size();
        const std::size_t bytes    = elements * sizeof(std::uint16_t);

        set_device(ec, 0);
        const DeviceBuffer parent_x = to_device_bf16(activation);
        std::array<DeviceBuffer, 2> shard_x;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            std::vector<float> block(static_cast<std::size_t>(sk) * tokens);
            for (std::int32_t token = 0; token < tokens; ++token) {
                const std::size_t source =
                    static_cast<std::size_t>(token) * k + rank * static_cast<std::size_t>(sk);
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

            set_device(ec, 0);
            GuardedDeviceBuffer reference(bytes);
            reference.copy_from_host(residual_bits.data(), bytes);
            DeviceArena reference_arena(
                std::max<std::size_t>(ops::linear_add_workspace_capacity_bytes(
                                          test_case.qtype, n, k, policy, tokens, tokens),
                                      1));
            const Tensor reference_x(parent_x.p, DType::BF16, {k, tokens});
            Tensor reference_residual(reference.data(), DType::BF16, {n, tokens});
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear_add(reference_x, parent_device.weight, reference_residual, policy,
                            reference_arena, ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += reference.verify_guards(label + " reference");
            const std::vector<double> expected = from_device_bf16(reference.data(), elements);

            const std::size_t workspace_bytes =
                std::max<std::size_t>(ops::linear_add_row_parallel_workspace_capacity_bytes(
                                          test_case.qtype, n, sk, policy, tokens, tokens),
                                      1);
            std::array<std::optional<GuardedDeviceBuffer>, 2> residual;
            std::array<std::optional<DeviceBuffer>, 2> staging;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                set_device(ec, static_cast<int>(rank));
                residual[rank].emplace(bytes);
                residual[rank]->copy_from_host(residual_bits.data(), bytes);
                staging[rank].emplace(bytes);
                arena[rank].emplace(workspace_bytes);
            }
            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {sk, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {sk, tokens})};
            const std::array<Weight, 2> w{shard[0].weight, shard[1].weight};
            const std::array<Tensor, 2> residual_view{
                Tensor(residual[0]->data(), DType::BF16, {n, tokens}),
                Tensor(residual[1]->data(), DType::BF16, {n, tokens})};
            const std::array<Tensor, 2> staging_view{
                Tensor(staging[0]->p, DType::BF16, {n, tokens}),
                Tensor(staging[1]->p, DType::BF16, {n, tokens})};
            const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::linear_add_row_parallel(x, w, residual_view, staging_view, policy, workspace, ec,
                                         events);
            synchronize_both(ec);

            const ReductionCriterion criterion = criterion_for(test_case, policy);
            std::array<std::vector<double>, 2> observed;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                const std::string rank_label = label + " rank " + std::to_string(rank);
                set_device(ec, static_cast<int>(rank));
                failures += residual[rank]->verify_guards(rank_label);
                observed[rank] = from_device_bf16(residual[rank]->data(), elements);
                failures += compare(rank_label, observed[rank], expected, criterion);
            }
            // The all-reduce combines the same operands on both ranks, so the sums are identical.
            if (observed[0] != observed[1]) {
                std::cerr << label << ": the ranks disagree after the all-reduce\n";
                ++failures;
            }
        }
    }
    return failures;
}

// linear_add_row_parallel_workspace_capacity_bytes() runs each format's shape resolvers without a
// device, so the registered halves are checked even where the parity cases must skip.
int verify_registry() {
    struct Entry {
        QType qtype;
        std::int32_t n;
        std::int32_t k;
        ops::LinearPolicy policy;
    };

    const std::vector<Entry> admitted{
        {QType::NVFP4, 5120, 8704, ops::LinearPolicy::A16Only},
        {QType::NVFP4, 5120, 8704, ops::LinearPolicy::AllowA4},
        {QType::NVFP4, 5120, 3072, ops::LinearPolicy::A16Only},
        {QType::NVFP4, 5120, 3072, ops::LinearPolicy::AllowA4},
        {QType::FP8_E4M3FN_ROW_BF16, 5120, 3072, ops::LinearPolicy::A16Only},
        {QType::FP8_E4M3FN_ROW_BF16, 5120, 3072, ops::LinearPolicy::AllowA8},
        {QType::FP8_E4M3FN_ROW_BF16, 5120, 8704, ops::LinearPolicy::A16Only},
        {QType::FP8_E4M3FN_ROW_BF16, 5120, 8704, ops::LinearPolicy::AllowA8},
    };
    int failures = 0;
    for (const Entry& entry : admitted) {
        for (const std::int32_t tokens : {1, 2, 8, 22, 25, 48, 1024}) {
            try {
                (void)ops::linear_add_row_parallel_workspace_capacity_bytes(
                    entry.qtype, entry.n, entry.k, entry.policy, tokens, tokens);
            } catch (const std::exception& error) {
                std::cerr << "registry: [" << entry.n << ',' << entry.k << "] qtype "
                          << static_cast<int>(entry.qtype) << ' ' << policy_name(entry.policy)
                          << " T=" << tokens << " rejected: " << error.what() << '\n';
                ++failures;
            }
        }
    }
    // Halves no format registers.
    const std::vector<Entry> rejected{
        {QType::BF16, 5120, 3072, ops::LinearPolicy::A16Only},
        {QType::Q5_G64_FP16, 5120, 8704, ops::LinearPolicy::A16Only},
    };
    for (const Entry& entry : rejected) {
        try {
            (void)ops::linear_add_row_parallel_workspace_capacity_bytes(
                entry.qtype, entry.n, entry.k, entry.policy, 1, 1);
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

    constexpr std::int32_t kN = 5120;
    constexpr std::int32_t kK = 8704;
    const std::size_t x_bytes = static_cast<std::size_t>(kK) * 2 * sizeof(std::uint16_t);
    const std::size_t y_bytes = static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t);
    set_device(ec, 0);
    DeviceBuffer x0(x_bytes);
    DeviceBuffer r0(y_bytes);
    DeviceBuffer stage0(y_bytes);
    set_device(ec, 1);
    DeviceBuffer x1(x_bytes);
    DeviceBuffer r1(y_bytes);
    DeviceBuffer stage1(y_bytes);

    Weight weight{};
    weight.qtype = QType::FP8_E4M3FN_ROW_BF16;
    weight.n     = kN;
    weight.k     = kK;

    expect_throw("token count", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 2}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> r{Tensor(r0.p, DType::BF16, {kN, 2}),
                                      Tensor(r1.p, DType::BF16, {kN, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 2}),
                                            Tensor(stage1.p, DType::BF16, {kN, 1})};
        ops::linear_add_row_parallel(x, {weight, weight}, r, staging, ec, events);
    });
    expect_throw("row N", [&] {
        Weight other = weight;
        other.n      = kN / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> r{Tensor(r0.p, DType::BF16, {kN, 1}),
                                      Tensor(r1.p, DType::BF16, {kN / 2, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 1}),
                                            Tensor(stage1.p, DType::BF16, {kN / 2, 1})};
        ops::linear_add_row_parallel(x, {weight, other}, r, staging, ec, events);
    });
    // The weight has no payload: a pair whose staging is rejected only by the all-reduce would
    // have dispatched both GEMMs first. The staging is checked before either rank issues work.
    expect_throw("staging shape", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> r{Tensor(r0.p, DType::BF16, {kN, 1}),
                                      Tensor(r1.p, DType::BF16, {kN, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 1}),
                                            Tensor(stage1.p, DType::BF16, {kN, 2})};
        ops::linear_add_row_parallel(x, {weight, weight}, r, staging, ec, events);
    });
    expect_throw("single-device context", [&] {
        const ExecutionContext single({ec.dev[0]->device});
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> r{Tensor(r0.p, DType::BF16, {kN, 1}),
                                      Tensor(r1.p, DType::BF16, {kN, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 1}),
                                            Tensor(stage1.p, DType::BF16, {kN, 1})};
        ops::linear_add_row_parallel(x, {weight, weight}, r, staging, single, events);
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
        std::cerr << "linear_add split registry: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cout << "FAIL linear_add split (registry)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: linear_add split parity requires two CUDA devices, found "
                  << device_count << '\n';
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
        // Token counts reach each half's decode, SIMT-chunk, A8/A4 crossover and MMA routes, the
        // W4A4 schedule seams, and (at 1024) the whole problem's TMA route.
        const std::vector<Case> cases{
            {"nvfp4 mlp down",
             QType::NVFP4,
             5120,
             17408,
             31U,
             {1, 7, 8, 48, 128, 384, 512, 1024},
             {kA16, kA4}},
            {"nvfp4 output",
             QType::NVFP4,
             5120,
             6144,
             34U,
             {1, 6, 7, 8, 32, 48, 128, 384, 512, 1024},
             {kA16, kA4}},
            {"fp8 output",
             QType::FP8_E4M3FN_ROW_BF16,
             5120,
             6144,
             32U,
             {1, 5, 21, 22, 24, 25, 48, 1024},
             {kA16, kA8}},
            {"fp8 mlp down",
             QType::FP8_E4M3FN_ROW_BF16,
             5120,
             17408,
             33U,
             {1, 8, 24, 25, 48, 128, 1024},
             {kA16, kA8}},
        };
        for (const Case& test_case : cases) { failures += run_case(test_case, ec, events); }
    } catch (const std::exception& error) {
        std::cerr << "linear_add split: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " linear_add split\n";
    return failures ? 1 : 0;
}
