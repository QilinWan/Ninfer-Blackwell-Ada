// A column of a fused residual projection must not change its bits with the batch width.
//
// The NVFP4 A4 and FP8 A8 contractions launch a FullTokens instance when the width fills whole
// token tiles and a predicated one otherwise. Their epilogue used to scale the accumulator outside
// the live-column predicate and add the residual inside it, so the compiler could contract the
// MUL+ADD into one FMA in the full instance but not in the predicated one: the same column then
// rounded differently in a full tile and in a tail tile, and a token's result depended on how many
// tokens shared its chunk.
//
// Every token column here carries the same input and the same residual, so every output column of
// every width must equal the columns of the narrowest A4/A8 width bit for bit. Each row holds an
// exact dot product (two nonzero products of short significands, exact in FP32 under any reduction
// order) that its residual nearly cancels, and rows differ, so the final BF16 store resolves the
// FP32 rounding of scale+add on many rows: one FMA versus a MUL then an ADD.
#include "core/weight.h"
#include "ninfer/ops/linear_add.h"
#include "core/device.h"

#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr std::int32_t kRows      = 5120;
constexpr std::int32_t kMaxTokens = 1025;
constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
// The cancelled rows carry tiny values; the uncancelled ones keep the tensor norm meaningful.
constexpr ReductionCriterion kA4Tolerance{0.16, kBf16UnitRoundoff, 0.16};
constexpr ReductionCriterion kA8Tolerance{0.04, kBf16UnitRoundoff, 0.06};

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

// Widths from the first A4/A8 width of the route to past the TMA floor: every token-tile schedule
// of the A4 MMA bands (M32 to 64, M32N128 to 128, M64 to 192, M128 above), each on a full and a
// ragged width, and the 1024/1025 TMA and split-K widths.
std::vector<std::int32_t> widths(std::int32_t first) {
    std::vector<std::int32_t> result;
    for (std::int32_t t = first; t <= 48; ++t) result.push_back(t);
    for (std::int32_t t : {49, 63, 64, 65, 96, 127, 128, 129, 191, 192, 193, 255, 256, 257, 383,
                           384, 385, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025}) {
        result.push_back(t);
    }
    return result;
}

struct Problem {
    std::string name;
    QType qtype;
    ops::LinearPolicy policy;
    std::int32_t k;
    std::int32_t first;
    quantized_weight::PackedWeight weight;
    std::vector<std::uint16_t> input;   // [kMaxTokens][k], every token identical
    std::vector<std::uint16_t> initial; // [kMaxTokens][kRows], every token identical
    std::vector<double> oracle;         // [kRows]: the represented projection plus the residual
    ReductionCriterion tolerance;
};

std::uint16_t cancelling_residual(std::int32_t row, double ideal) {
    // Odd rows keep a zero residual so the tensor-level relative criterion has a reference norm.
    return row % 2 ? f32_to_bf16(0.0F) : f32_to_bf16(-static_cast<float>(ideal));
}

void broadcast_columns(Problem& problem, const std::vector<std::uint16_t>& column,
                       const std::vector<std::uint16_t>& residual) {
    problem.input.assign(static_cast<std::size_t>(problem.k) * kMaxTokens, 0);
    problem.initial.assign(static_cast<std::size_t>(kRows) * kMaxTokens, 0);
    for (std::int32_t token = 0; token < kMaxTokens; ++token) {
        std::copy(column.begin(), column.end(),
                  problem.input.begin() + static_cast<std::size_t>(token) * problem.k);
        std::copy(residual.begin(), residual.end(),
                  problem.initial.begin() + static_cast<std::size_t>(token) * kRows);
    }
}

// NVFP4: E2M1 codes at columns 0 and 16 (two K16 groups) with per-row E4M3 group scales, input 6.0
// in both groups (block scale 1.0 with input divisor 1, so A4 represents it exactly), and the
// global weight divisor 0.13 that makes alpha inexact.
Problem make_nvfp4(std::int32_t k, std::uint32_t seed) {
    Problem problem;
    problem.name   = "NVFP4 A4 [" + std::to_string(kRows) + "," + std::to_string(k) + "]";
    problem.qtype  = QType::NVFP4;
    problem.policy = ops::LinearPolicy::AllowA4;
    problem.k      = k;
    problem.first  = k == 6144 || k == 3072 ? ops::detail::kNvfp4OutputFamilyFirstA4Tokens
                                            : ops::detail::kNvfp4DownFamilyFirstA4Tokens;
    problem.tolerance = kA4Tolerance;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.13F;
    options.input_scale_divisor  = 1.0F;
    problem.weight = quantized_weight::make_patterned_weight(QType::NVFP4, kRows, k, seed, options);
    auto& packed   = problem.weight;
    std::fill_n(packed.payload.begin(), static_cast<std::size_t>(kRows) * k / 2, 0);
    std::fill_n(packed.payload.begin() + static_cast<std::ptrdiff_t>(packed.scale_plane_offset),
                packed.scale_plane_bytes, 0);
    for (std::int32_t row = 0; row < kRows; ++row) {
        const std::uint32_t h = mix(static_cast<std::uint32_t>(row) ^ seed);
        const std::size_t codes = static_cast<std::size_t>(row) * k / 2;
        packed.payload[codes]     = static_cast<std::uint8_t>(1 + h % 7);        // column 0
        packed.payload[codes + 8] = static_cast<std::uint8_t>(1 + (h >> 3) % 7); // column 16
        const std::size_t scales = packed.scale_plane_offset +
                                   static_cast<std::size_t>(row / 128) * (k / 64) * 512 +
                                   static_cast<std::size_t>(row % 32) * 16 +
                                   static_cast<std::size_t>((row % 128) / 32) * 4;
        packed.payload[scales]     = static_cast<std::uint8_t>(0x28 + (h >> 6) % 32);  // group 0
        packed.payload[scales + 1] = static_cast<std::uint8_t>(0x28 + (h >> 11) % 32); // group 1
    }
    std::vector<std::uint16_t> column(k, 0);
    column[0] = column[16] = f32_to_bf16(6.0F);
    std::vector<std::uint16_t> residual(kRows);
    problem.oracle.resize(kRows);
    for (std::int32_t row = 0; row < kRows; ++row) {
        const double ideal = 6.0 * quantized_weight::logical_weight_fp64(packed, row, 0) +
                             6.0 * quantized_weight::logical_weight_fp64(packed, row, 16);
        residual[row]       = cancelling_residual(row, ideal);
        problem.oracle[row] = ideal + static_cast<double>(bf16_to_f32(residual[row]));
    }
    broadcast_columns(problem, column, residual);
    return problem;
}

// FP8: E4M3 codes at columns 0 and 1 with per-row BF16 scales, input 1.0 and 0.30078125. A8 takes
// the token scale 1/448 and stores the codes 448 and 128, so the dot product is exact and the
// activation scale, the row scale and the residual all enter the epilogue.
Problem make_fp8(std::int32_t k, std::uint32_t seed) {
    Problem problem;
    problem.name   = "FP8 A8 [" + std::to_string(kRows) + "," + std::to_string(k) + "]";
    problem.qtype  = QType::FP8_E4M3FN_ROW_BF16;
    problem.policy = ops::LinearPolicy::AllowA8;
    problem.k      = k;
    problem.first  = k == 6144 || k == 3072 ? ops::detail::kFp8OutputFamilyFirstA8Tokens
                                            : ops::detail::kFp8DownFamilyFirstA8Tokens;
    problem.tolerance = kA8Tolerance;
    problem.weight =
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kRows, k, seed);
    auto& packed = problem.weight;
    std::fill_n(packed.payload.begin(), packed.code_plane_bytes, 0);
    for (std::int32_t row = 0; row < kRows; ++row) {
        const std::uint32_t h   = mix(static_cast<std::uint32_t>(row) ^ seed);
        const std::size_t codes = static_cast<std::size_t>(row) * k;
        packed.payload[codes]     = static_cast<std::uint8_t>(0x30 + h % 16);
        packed.payload[codes + 1] = static_cast<std::uint8_t>(0x30 + (h >> 4) % 24);
        const float scale = std::ldexp(1.0F + static_cast<float>((h >> 9) % 128) / 128.0F,
                                       -6 - static_cast<int>((h >> 16) % 4));
        quantized_weight::detail::store_u16_le(
            packed.payload, packed.scale_plane_offset + static_cast<std::size_t>(row) * 2,
            f32_to_bf16(scale));
    }
    std::vector<std::uint16_t> column(k, 0);
    column[0] = f32_to_bf16(1.0F);
    column[1] = f32_to_bf16(0.30078125F);
    const double token_scale = static_cast<double>(1.0F / 448.0F);
    std::vector<std::uint16_t> residual(kRows);
    problem.oracle.resize(kRows);
    for (std::int32_t row = 0; row < kRows; ++row) {
        const double ideal = (448.0 * quantized_weight::logical_weight_fp64(packed, row, 0) +
                              128.0 * quantized_weight::logical_weight_fp64(packed, row, 1)) *
                             token_scale;
        residual[row]       = cancelling_residual(row, ideal);
        problem.oracle[row] = ideal + static_cast<double>(bf16_to_f32(residual[row]));
    }
    broadcast_columns(problem, column, residual);
    return problem;
}

std::string hex(std::uint16_t bits) {
    char text[8];
    std::snprintf(text, sizeof(text), "0x%04x", bits);
    return text;
}

int run(Problem& problem) {
    GuardedDeviceBuffer device_weight(problem.weight.payload.size());
    device_weight.copy_from_host(problem.weight.payload.data(), problem.weight.payload.size());
    const Weight weight = problem.weight.device_weight(device_weight.data());
    GuardedDeviceBuffer device_input(problem.input.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(problem.input.data(), device_input.bytes());

    int failures = 0, changed = 0;
    const std::vector<std::int32_t> tested = widths(problem.first);
    std::vector<std::uint16_t> reference; // the columns of the first width
    for (const std::int32_t t : tested) {
        const std::size_t words = static_cast<std::size_t>(kRows) * t;
        GuardedDeviceBuffer output(words * sizeof(std::uint16_t));
        output.copy_from_host(problem.initial.data(), output.bytes());
        Tensor x(device_input.data(), DType::BF16, {problem.k, t});
        Tensor residual(output.data(), DType::BF16, {kRows, t});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            problem.qtype, kRows, problem.k, problem.policy, t, t);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, problem.policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize residual rounding linear_add");
        std::vector<std::uint16_t> actual(words);
        output.copy_to_host(actual.data(), output.bytes());
        const std::string label = problem.name + " T=" + std::to_string(t);
        failures += output.verify_guards(label);

        if (t == tested.front()) { reference.assign(actual.begin(), actual.begin() + kRows); }
        std::size_t differing = 0, first = words;
        for (std::size_t i = 0; i < words; ++i) {
            if (actual[i] != reference[i % kRows]) {
                if (differing++ == 0) first = i;
            }
        }
        if (differing != 0) {
            const std::size_t row = first % kRows, token = first / kRows;
            std::cerr << label << ": " << differing << " of " << words
                      << " values differ from the T=" << tested.front() << " columns (first: row "
                      << row << " token " << token << " " << hex(actual[first]) << ", T="
                      << tested.front() << " gave " << hex(reference[row]) << ")\n";
            ++changed;
            ++failures;
        }
        std::vector<double> observed(kRows);
        for (std::int32_t row = 0; row < kRows; ++row) observed[row] = bf16_to_f32(actual[row]);
        failures += verify_reduction(label, observed, problem.oracle, problem.tolerance);
    }
    failures += device_input.verify_guards(problem.name + " input");
    failures += device_weight.verify_guards(problem.name + " weight");
    std::cout << problem.name << ": " << tested.size() << " widths " << tested.front() << ".."
              << tested.back() << ", "
              << (changed == 0 ? std::string("every column bit-identical")
                               : std::to_string(changed) + " widths changed bits")
              << '\n';
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // The full problems and the two-device input-column halves, whose fused rank-0 residual update
    // runs the same contractions at tp 2.
    for (const std::int32_t k : {6144, 17408, 3072, 8704}) {
        Problem nvfp4 = make_nvfp4(k, 911U + static_cast<std::uint32_t>(k));
        failures += run(nvfp4);
    }
    for (const std::int32_t k : {6144, 17408, 3072, 8704}) {
        Problem fp8 = make_fp8(k, 919U + static_cast<std::uint32_t>(k));
        failures += run(fp8);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " linear_add residual rounding\n";
    return failures == 0 ? 0 : 1;
}
