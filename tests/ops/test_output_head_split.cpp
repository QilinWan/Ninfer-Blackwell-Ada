// Two-device parity of the vocabulary-split output head.
//
// The FP8 output head `[248320,5120]` splits its output rows, the vocabulary, into two
// `[124160,5120]` halves. Each rank computes its half of the logits with linear_column_parallel,
// and allgather_rows then rebuilds the whole `[248320,T]` logits on both devices. This suite
// compares each half and each rebuilt image against linear() over the whole head on device 0.
// linear() stores the vocabulary fastest, so the gather runs once per token, over `[1,124160]`
// blocks.
//
// The whole head and its halves are generated independently from one logical matrix, and each case
// first checks that each half is the parent's block and that the two halves differ. Every policy
// keeps A16 compute for the vocabulary head, so the halves resolve the whole head's kernels at the
// halved extent.
//
// Every case needs two CUDA devices in one process and the suite reports 77 with fewer. The
// registry probe is host-only and runs first.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "core/weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr double kBf16Ulp      = 1.0 / 256.0;
constexpr std::int32_t kVocab  = 248320;
constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kHalf   = kVocab / 2;
constexpr QType kQType         = QType::FP8_E4M3FN_ROW_BF16;
constexpr ReductionCriterion kHeadCriterion{2.0 * kBf16Ulp, 0.0, 2.0 * kBf16Ulp};

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

void* element_offset(void* base, std::size_t elements) {
    return static_cast<std::uint8_t*>(base) + elements * sizeof(std::uint16_t);
}

qw::PackedWeight make_head(std::int32_t rows, std::uint32_t seed, std::int32_t row_origin) {
    qw::PatternedWeightOptions options;
    options.row_origin              = row_origin;
    options.decorrelate_coordinates = true;
    return qw::make_patterned_weight(kQType, rows, kHidden, seed, options);
}

int verify_half_is_parent_block(const std::string& label, const qw::PackedWeight& parent,
                                const qw::PackedWeight& half, std::int32_t row_origin) {
    for (const std::int32_t row : {0, 1, 127, 128, kHalf / 2, kHalf - 1}) {
        for (const std::int32_t column : {0, 1, 511, 512, kHidden / 2, kHidden - 1}) {
            const double got      = qw::logical_weight_fp64(half, row, column);
            const double expected = qw::logical_weight_fp64(parent, row + row_origin, column);
            if (got != expected) {
                std::cerr << label << ": half (" << row << ',' << column
                          << ") is not the parent's: " << got << " vs " << expected << '\n';
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

std::size_t workspace_bytes(std::int32_t rows, ops::LinearPolicy policy, std::int32_t tokens) {
    return std::max<std::size_t>(
        ops::linear_workspace_capacity_bytes(kQType, rows, kHidden, policy, tokens, tokens), 1);
}

int compare(const std::string& label, const std::vector<double>& got,
            const std::vector<double>& expected) {
    const ReductionStats stats =
        compute_reduction_stats(got.data(), expected.data(), static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2 << '\n';
    return verify_reduction(label, got, expected, kHeadCriterion);
}

int run_head(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr std::uint32_t kSeed = 301U;
    int failures                  = 0;
    const qw::PackedWeight parent = make_head(kVocab, kSeed, 0);
    const std::array<qw::PackedWeight, 2> half{make_head(kHalf, kSeed, 0),
                                               make_head(kHalf, kSeed, kHalf)};
    failures += verify_half_is_parent_block("output head half 0", parent, half[0], 0);
    failures += verify_half_is_parent_block("output head half 1", parent, half[1], kHalf);
    if (half[0].payload == half[1].payload) {
        std::cerr << "output head: the two halves are byte-identical\n";
        ++failures;
    }
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    const RankWeight parent_weight = upload(parent);
    std::array<RankWeight, 2> half_weight;
    for (std::size_t rank = 0; rank < 2; ++rank) {
        set_device(ec, static_cast<int>(rank));
        half_weight[rank] = upload(half[rank]);
    }

    constexpr std::array policies{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                  ops::LinearPolicy::AllowA4};
    // T=1 is decode; 41/42 and 48/49 cross the K-split and MMA boundaries, and 49 also chunks.
    for (const std::int32_t tokens : {1, 8, 41, 42, 48, 49}) {
        std::vector<float> activation(static_cast<std::size_t>(kHidden) * tokens);
        fill_uniform(activation, kSeed * 31U + static_cast<std::uint32_t>(tokens), -1.0F, 1.0F);
        round_to_bf16(activation);
        std::array<DeviceBuffer, 2> x_device;
        for (std::size_t rank = 0; rank < 2; ++rank) {
            set_device(ec, static_cast<int>(rank));
            x_device[rank] = to_device_bf16(activation);
        }

        for (const ops::LinearPolicy policy : policies) {
            const std::string label =
                "fp8 output head T=" + std::to_string(tokens) + " " + policy_name(policy);
            const std::size_t whole_elements = static_cast<std::size_t>(kVocab) * tokens;
            const std::size_t half_elements  = static_cast<std::size_t>(kHalf) * tokens;

            set_device(ec, 0);
            GuardedDeviceBuffer reference(whole_elements * sizeof(std::uint16_t));
            reference.fill(0xff);
            DeviceArena reference_arena(workspace_bytes(kVocab, policy, tokens));
            const Tensor reference_x(x_device[0].p, DType::BF16, {kHidden, tokens});
            Tensor reference_out(reference.data(), DType::BF16, {kVocab, tokens});
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear(reference_x, parent_weight.weight, reference_out, policy, reference_arena,
                        ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += reference.verify_guards(label + " reference");
            const std::vector<double> expected = from_device_bf16(reference.data(), whole_elements);

            std::array<std::optional<GuardedDeviceBuffer>, 2> half_out;
            std::array<std::optional<GuardedDeviceBuffer>, 2> logits;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                set_device(ec, static_cast<int>(rank));
                half_out[rank].emplace(half_elements * sizeof(std::uint16_t));
                half_out[rank]->fill(0xff);
                logits[rank].emplace(whole_elements * sizeof(std::uint16_t));
                logits[rank]->fill(0xcd);
                arena[rank].emplace(workspace_bytes(kHalf, policy, tokens));
            }
            const std::array<Tensor, 2> x{Tensor(x_device[0].p, DType::BF16, {kHidden, tokens}),
                                          Tensor(x_device[1].p, DType::BF16, {kHidden, tokens})};
            const std::array<Weight, 2> w{half_weight[0].weight, half_weight[1].weight};
            const std::array<Tensor, 2> out{
                Tensor(half_out[0]->data(), DType::BF16, {kHalf, tokens}),
                Tensor(half_out[1]->data(), DType::BF16, {kHalf, tokens})};
            const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::linear_column_parallel(x, w, out, policy, workspace, ec);
            for (std::int32_t token = 0; token < tokens; ++token) {
                const auto half_offset  = static_cast<std::size_t>(token) * kHalf;
                const auto whole_offset = static_cast<std::size_t>(token) * kVocab;
                const std::array<Tensor, 2> part{
                    Tensor(element_offset(half_out[0]->data(), half_offset), DType::BF16,
                           {1, kHalf}),
                    Tensor(element_offset(half_out[1]->data(), half_offset), DType::BF16,
                           {1, kHalf})};
                const std::array<Tensor, 2> destination{
                    Tensor(element_offset(logits[0]->data(), whole_offset), DType::BF16,
                           {1, kVocab}),
                    Tensor(element_offset(logits[1]->data(), whole_offset), DType::BF16,
                           {1, kVocab})};
                ops::allgather_rows(destination, part, ec, events);
            }
            synchronize_both(ec);

            std::array<std::vector<double>, 2> half_observed;
            std::array<std::vector<double>, 2> rebuilt;
            for (std::size_t rank = 0; rank < 2; ++rank) {
                const std::string rank_label = label + " rank " + std::to_string(rank);
                set_device(ec, static_cast<int>(rank));
                failures += half_out[rank]->verify_guards(rank_label);
                failures += logits[rank]->verify_guards(rank_label + " gathered");
                half_observed[rank] = from_device_bf16(half_out[rank]->data(), half_elements);
                rebuilt[rank]       = from_device_bf16(logits[rank]->data(), whole_elements);

                std::vector<double> block(half_elements);
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const std::size_t source = static_cast<std::size_t>(token) * kVocab +
                                               rank * static_cast<std::size_t>(kHalf);
                    std::copy(expected.begin() + static_cast<std::ptrdiff_t>(source),
                              expected.begin() + static_cast<std::ptrdiff_t>(source + kHalf),
                              block.begin() + static_cast<std::ptrdiff_t>(token) * kHalf);
                }
                failures += compare(rank_label, half_observed[rank], block);
                failures += compare(rank_label + " gathered", rebuilt[rank], expected);
            }
            if (half_observed[0] == half_observed[1]) {
                std::cerr << label << ": both ranks produced the same logit half\n";
                ++failures;
            }
            // The gather relocates bytes, so both devices hold the identical image.
            if (rebuilt[0] != rebuilt[1]) {
                std::cerr << label << ": the gathered logits differ between devices\n";
                ++failures;
            }
        }
    }
    return failures;
}

int verify_registry() {
    int failures = 0;
    for (const ops::LinearPolicy policy :
         {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
        try {
            if (ops::linear_workspace_capacity_bytes(kQType, kHalf, kHidden, policy, 1, 2048) !=
                0) {
                std::cerr << "registry: the FP8 head half reported nonzero A16 workspace\n";
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << "registry: the FP8 head half was rejected for " << policy_name(policy)
                      << ": " << error.what() << '\n';
            ++failures;
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " registry: FP8 head half\n";
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    try {
        failures = verify_registry();
    } catch (const std::exception& error) {
        std::cerr << "output head split registry: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cout << "FAIL output head split (registry)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: output head split parity requires two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    try {
        const ExecutionContext ec({0, 1});
        std::cout << "peer access: " << (ops::enable_peer_access(ec) ? "direct" : "host-staged")
                  << '\n';
        const ops::PeerEvents events(ec);
        failures += run_head(ec, events);
    } catch (const std::exception& error) {
        std::cerr << "output head split: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " output head split\n";
    return failures ? 1 : 0;
}
