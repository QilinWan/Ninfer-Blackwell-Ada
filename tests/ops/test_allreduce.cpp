// Two-device qualification suite for the cross-device collectives declared in
// include/ninfer/ops/allreduce.h.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing.
//
// Oracles:
//   allreduce_sum - independent FP64 elementwise sum of the two represented BF16 inputs. The
//     Op's observable output is BF16, so the comparison allows one BF16 ulp of output storage
//     rounding (relative 3.95e-3 >= 2^-8); the oracle itself performs no rounding. The two ranks'
//     results must also be bit-identical.
//   allgather_rows - exact: the Op only relocates rows, so every destination byte is compared
//     bit-for-bit against the concatenated source halves.
//
// Both transports are qualified: the staged path eagerly and inside a two-device CUDA Graph, and
// the PeerMailbox exchange inside a graph replayed several times with fresh inputs.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ops/op_tester.h"
#include "ops/split_test_support.h"

#include "core/decode_graph.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One BF16 ulp (2^-8 = 3.90625e-3) with the same small margin the residual_add suite uses. The
// only error the Op may introduce is the single round-to-nearest-even of a+b into BF16 storage,
// which is bounded by half an ulp; the criterion is deliberately not tighter than one ulp so a
// double-rounded tie cannot make the suite flaky.
constexpr PointwiseCriterion allreduce_sum_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Independent oracle: the complete formula in FP64 from the represented BF16 inputs.
std::vector<double> allreduce_sum_oracle(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<double> expected(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        expected[i] = static_cast<double>(a[i]) + static_cast<double>(b[i]);
    }
    return expected;
}

// `ne0` is the contiguous dimension and `ne1` the outer one, so a 1-D buffer passes ne1 == 1 and
// the real row-parallel residual passes {5120, 48}.
int run_allreduce_case(const char* label, std::int32_t ne0, std::int32_t ne1, std::uint32_t seed,
                       const ExecutionContext& ec, const ops::PeerEvents& events) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    std::vector<float> a(count), b(count);
    fill_uniform(a, seed, -8.0f, 8.0f);
    fill_uniform(b, seed + 1, -8.0f, 8.0f);
    round_to_bf16(a);
    round_to_bf16(b);

    const auto expected     = allreduce_sum_oracle(a, b);
    const auto a_bits       = encode_bf16(a);
    const auto b_bits       = encode_bf16(b);
    const std::size_t bytes = count * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.copy_from_host(a_bits.data(), bytes);
    staging_0.fill(0);

    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.copy_from_host(b_bits.data(), bytes);
    staging_1.fill(0);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {ne0, ne1}),
                                       Tensor(buffer_1.data(), DType::BF16, {ne0, ne1})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {ne0, ne1}),
                                        Tensor(staging_1.data(), DType::BF16, {ne0, ne1})};

    retire_staging(ec);
    ops::allreduce_sum(buffer, staging, ec, events);
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures += verify_pointwise((std::string(label) + " device 0").c_str(),
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("allreduce buffer device 0");
    failures += staging_0.verify_guards("allreduce staging device 0");
    set_device(ec, 1);
    failures += verify_pointwise((std::string(label) + " device 1").c_str(),
                                 from_device_bf16(buffer_1.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("allreduce buffer device 1");
    failures += staging_1.verify_guards("allreduce staging device 1");
    return failures;
}

// `row_length` is ne[0] (the contiguous row) and the gathered axis is ne[1], so device 0 owns the
// leading `rows_0` rows and device 1 the trailing `rows_1`.
int run_allgather_case(const char* label, std::int32_t rows_0, std::int32_t rows_1,
                       std::int32_t row_length, std::uint32_t seed, const ExecutionContext& ec,
                       const ops::PeerEvents& events) {
    const std::int32_t rows  = rows_0 + rows_1;
    const std::size_t part_0 = static_cast<std::size_t>(rows_0) * row_length;
    const std::size_t part_1 = static_cast<std::size_t>(rows_1) * row_length;

    std::vector<float> source_0(part_0), source_1(part_1);
    fill_uniform(source_0, seed, -8.0f, 8.0f);
    fill_uniform(source_1, seed + 1, -8.0f, 8.0f);
    const auto bits_0 = encode_bf16(source_0);
    const auto bits_1 = encode_bf16(source_1);

    // Exact oracle: the gathered image is the concatenation of the two owned row ranges.
    std::vector<std::uint16_t> expected;
    expected.reserve(part_0 + part_1);
    expected.insert(expected.end(), bits_0.begin(), bits_0.end());
    expected.insert(expected.end(), bits_1.begin(), bits_1.end());

    const std::size_t full_bytes = expected.size() * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer source_device_0(part_0 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_0(full_bytes);
    source_device_0.copy_from_host(bits_0.data(), source_device_0.bytes());
    destination_0.fill(0xcd);

    set_device(ec, 1);
    GuardedDeviceBuffer source_device_1(part_1 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_1(full_bytes);
    source_device_1.copy_from_host(bits_1.data(), source_device_1.bytes());
    destination_1.fill(0xcd);

    const std::array<Tensor, 2> destination{
        Tensor(destination_0.data(), DType::BF16, {row_length, rows}),
        Tensor(destination_1.data(), DType::BF16, {row_length, rows})};
    const std::array<Tensor, 2> part{
        Tensor(source_device_0.data(), DType::BF16, {row_length, rows_0}),
        Tensor(source_device_1.data(), DType::BF16, {row_length, rows_1})};

    retire_staging(ec);
    ops::allgather_rows(destination, part, ec, events);
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures +=
        verify_exact((std::string(label) + " device 0").c_str(),
                     from_device<std::uint16_t>(destination_0.data(), expected.size()), expected);
    failures += verify_exact("allgather source device 0 unchanged",
                             from_device<std::uint16_t>(source_device_0.data(), part_0), bits_0);
    failures += destination_0.verify_guards("allgather destination device 0");
    failures += source_device_0.verify_guards("allgather source device 0");
    set_device(ec, 1);
    failures +=
        verify_exact((std::string(label) + " device 1").c_str(),
                     from_device<std::uint16_t>(destination_1.data(), expected.size()), expected);
    failures += verify_exact("allgather source device 1 unchanged",
                             from_device<std::uint16_t>(source_device_1.data(), part_1), bits_1);
    failures += destination_1.verify_guards("allgather destination device 1");
    failures += source_device_1.verify_guards("allgather source device 1");
    return failures;
}

// Regression guard for the cross-call write-after-read hazard.
//
// kRounds alternating collectives are issued back-to-back on ONE set of buffers, staging, and
// events with NO host synchronization until the very end -- exactly what the tensor-parallel
// forward loop produces (128 all-reduces per decode token, plus one logit allgather per column)
// and what a captured graph replays. Under the earlier push-based
// design, call k+1's inbound write into the peer's staging was ordered only against the sender's
// own stream history, not against the receiver's call-k read of that staging; this loop is the
// pattern where that window opens.
//
// Two properties make a stale operand observable rather than benign:
//   * the chain value changes every round -- both devices start at 2^-40 and each all-reduce
//     doubles it, so after k rounds every element is exactly 2^(k-40). Every intermediate is a
//     power of two, hence exact in BF16 and in the FP32 accumulator, and an operand from the wrong
//     round yields a value that is not the expected power of two (3v instead of 2v, say);
//   * the interleaved all-gather reads the live all-reduce buffers as its parts, so it also
//     carries the round's value into its destination and is checked against the same expectation,
//     and its own cross-call ordering (the peer must finish reading part[r] before the next
//     all-reduce overwrites it) is exercised rather than merely occupying the streams.
//
// Deliberate skew: a large memset is queued on rank 0's stream first, with no host sync, so the
// two streams run genuinely out of step for the first rounds instead of in lockstep.
int run_chained_case(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr std::int32_t n         = 5120;
    constexpr int kRounds            = 64;
    constexpr int kStartExponent     = -40;
    constexpr std::size_t kSkewBytes = 256u << 20;
    constexpr int kSkewRepeats       = 8;

    const float start           = std::ldexp(1.0f, kStartExponent);
    const double expected_value = std::ldexp(1.0, kStartExponent + kRounds);
    const std::size_t count     = static_cast<std::size_t>(n);
    const std::size_t bytes     = count * sizeof(std::uint16_t);

    const std::vector<std::uint16_t> start_bits(count, f32_to_bf16(start));
    const std::vector<double> expected(count, expected_value);
    // Both halves of the gathered image come from the two all-reduce buffers, which hold the same
    // value once the round's all-reduce has run.
    const std::vector<double> expected_gathered(2 * count, expected_value);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes), gathered_0(2 * bytes);
    buffer_0.copy_from_host(start_bits.data(), bytes);
    staging_0.fill(0);
    gathered_0.fill(0xcd);
    DeviceBuffer skew(kSkewBytes);

    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes), gathered_1(2 * bytes);
    buffer_1.copy_from_host(start_bits.data(), bytes);
    staging_1.fill(0);
    gathered_1.fill(0xcd);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> gathered{Tensor(gathered_0.data(), DType::BF16, {n, 2}),
                                         Tensor(gathered_1.data(), DType::BF16, {n, 2})};
    const std::array<Tensor, 2> part{Tensor(buffer_0.data(), DType::BF16, {n, 1}),
                                     Tensor(buffer_1.data(), DType::BF16, {n, 1})};

    retire_staging(ec);

    set_device(ec, 0);
    for (int i = 0; i < kSkewRepeats; ++i) {
        cuda_check(cudaMemsetAsync(skew.p, i & 0xff, skew.bytes, ec.dev[0]->stream),
                   "cudaMemsetAsync skew");
    }

    for (int round = 0; round < kRounds; ++round) {
        ops::allreduce_sum(buffer, staging, ec, events);
        ops::allgather_rows(gathered, part, ec, events);
    }
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures +=
        verify_pointwise("chained 64x device 0 buffer", from_device_bf16(buffer_0.data(), count),
                         expected, allreduce_sum_bf16_criterion());
    failures += verify_pointwise("chained 64x device 0 gathered",
                                 from_device_bf16(gathered_0.data(), 2 * count), expected_gathered,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("chained buffer device 0");
    failures += staging_0.verify_guards("chained staging device 0");
    failures += gathered_0.verify_guards("chained gathered device 0");
    set_device(ec, 1);
    failures +=
        verify_pointwise("chained 64x device 1 buffer", from_device_bf16(buffer_1.data(), count),
                         expected, allreduce_sum_bf16_criterion());
    failures += verify_pointwise("chained 64x device 1 gathered",
                                 from_device_bf16(gathered_1.data(), 2 * count), expected_gathered,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("chained buffer device 1");
    failures += staging_1.verify_guards("chained staging device 1");
    failures += gathered_1.verify_guards("chained gathered device 1");
    return failures;
}

// The per-decode-token all-reduce payload is one 5120-element BF16 hidden row = 10 KiB, issued
// 128 times per token. Measure the full call: both peer transfers plus both local combines, with
// the host waiting for both devices to retire the work.
//
// The figure is SYNC-DOMINATED and is an upper bound, not the op's device cost: each iteration
// pays two host-side cudaStreamSynchronize round trips that the production path does not, because
// the forward loop issues 128 of these back-to-back and the decode path replays them inside a
// captured CUDA graph with no host sync at all. Read it as "one isolated, fully drained
// all-reduce". It is informative only: host-sync latency depends on the platform and its load.
int run_microbenchmark(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr std::int32_t n        = 5120;
    constexpr int kWarmupIterations = 50;
    constexpr int kTimedIterations  = 500;
    const std::size_t bytes         = static_cast<std::size_t>(n) * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.fill(0);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.fill(0);
    staging_1.fill(0);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};

    retire_staging(ec);
    for (int i = 0; i < kWarmupIterations; ++i) {
        ops::allreduce_sum(buffer, staging, ec, events);
        synchronize_both(ec);
    }

    std::vector<double> samples;
    samples.reserve(kTimedIterations);
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < kTimedIterations; ++i) {
        const auto call_started = std::chrono::steady_clock::now();
        ops::allreduce_sum(buffer, staging, ec, events);
        synchronize_both(ec);
        const std::chrono::duration<double, std::micro> call_elapsed =
            std::chrono::steady_clock::now() - call_started;
        samples.push_back(call_elapsed.count());
    }
    const std::chrono::duration<double, std::micro> elapsed =
        std::chrono::steady_clock::now() - started;
    const double mean_micros = elapsed.count() / kTimedIterations;
    std::sort(samples.begin(), samples.end());

    std::cout << "allreduce microbench: " << bytes << " B bf16 mean " << mean_micros << " us, p50 "
              << samples[samples.size() / 2] << " us, p99 " << samples[(samples.size() * 99) / 100]
              << " us, max " << samples.back() << " us over " << kTimedIterations
              << " iterations (eager staged, host-sync dominated)\n";

    int failures = 0;
    set_device(ec, 0);
    failures += buffer_0.verify_guards("microbench buffer device 0");
    failures += staging_0.verify_guards("microbench staging device 0");
    set_device(ec, 1);
    failures += buffer_1.verify_guards("microbench buffer device 1");
    failures += staging_1.verify_guards("microbench staging device 1");
    return failures;
}

// Captures `body` into ONE two-device graph: rank 1's stream is forked into rank 0's capture.
void capture_two_devices(const ExecutionContext& ec, const DecodeGraphPeerBridge& bridge,
                         DecodeGraphDefinition& definition, const std::function<void()>& body) {
    set_device(ec, 0);
    definition.capture(
        ec.dev[0]->stream,
        [&] {
            body();
            set_device(ec, 0);
        },
        DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
}

void launch_two_devices(const ExecutionContext& ec, DecodeGraphExecutable& executable) {
    set_device(ec, 0);
    executable.launch(ec.dev[0]->stream);
    synchronize_both(ec);
}

// Captured replay, the decode route. `sites` all-reduces of one [ne0, ne1] buffer per rank are
// captured once into a two-device graph and the executable is launched kReplays times, each with
// fresh random inputs and no host step between launches other than the input upload. After the
// first site both ranks hold bf16(a + b); every later site doubles it exactly, so the oracle is
// 2^(sites-1) * (a + b) in FP64 with the one-ulp criterion.
//
// Stale operands are made observable: a captured memset of kSkewBytes on rank 1 runs before the
// first site, so rank 0 reaches every exchange long before rank 1 has republished. An exchange
// that accepted the previous replay's publication (the ABA hazard of a flag that is not reset or
// advanced between launches) would combine the previous inputs and miss the oracle.
//
// kBackToBack further launches are then issued with no host synchronization between them, as a
// serving loop may, and must double the value 'sites' times each. One eager call afterwards
// checks that the staged path composes with the captured history on the same buffers and events.
//
// `mailbox` names the mailbox attached to `events`, or null. The node count proves the selected
// transport: the mailbox exchange is one kernel node per device per site, the staged path at
// least two per device per site.
int run_captured_case(const char* label, std::int32_t ne0, std::int32_t ne1, int sites,
                      bool expect_mailbox, const ExecutionContext& ec,
                      const ops::PeerEvents& events, const ops::PeerMailbox* mailbox) {
    constexpr int kReplays           = 4;
    constexpr int kBackToBack        = 3;
    constexpr std::size_t kSkewBytes = 64u << 20;

    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    const std::size_t bytes = count * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    staging_1.fill(0);
    DeviceBuffer skew(kSkewBytes);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {ne0, ne1}),
                                       Tensor(buffer_1.data(), DType::BF16, {ne0, ne1})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {ne0, ne1}),
                                        Tensor(staging_1.data(), DType::BF16, {ne0, ne1})};
    retire_staging(ec);

    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        set_device(ec, 1);
        cuda_check(cudaMemsetAsync(skew.p, 0, skew.bytes, ec.dev[1]->stream), "skew memset");
        for (int site = 0; site < sites; ++site) {
            ops::allreduce_sum(buffer, staging, ec, events);
        }
    });

    int failures                = 0;
    const std::size_t nodes     = definition.node_count();
    const std::size_t exchanges = 1 + 2 * static_cast<std::size_t>(sites);
    if (expect_mailbox ? nodes != exchanges : nodes <= exchanges) {
        std::cerr << label << ": " << nodes << " graph nodes do not match the "
                  << (expect_mailbox ? "mailbox" : "staged") << " transport\n";
        ++failures;
    }

    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    const double scale = std::ldexp(1.0, sites - 1);
    std::vector<float> a(count), b(count);
    std::vector<double> expected(count);
    for (int replay = 0; replay < kReplays; ++replay) {
        const std::uint32_t seed = 301u + 2u * static_cast<std::uint32_t>(replay);
        fill_uniform(a, seed, -8.0f, 8.0f);
        fill_uniform(b, seed + 1, -8.0f, 8.0f);
        round_to_bf16(a);
        round_to_bf16(b);
        const auto sum = allreduce_sum_oracle(a, b);
        for (std::size_t i = 0; i < count; ++i) { expected[i] = scale * sum[i]; }
        const auto a_bits = encode_bf16(a);
        const auto b_bits = encode_bf16(b);
        set_device(ec, 0);
        buffer_0.copy_from_host(a_bits.data(), bytes);
        set_device(ec, 1);
        buffer_1.copy_from_host(b_bits.data(), bytes);
        retire_staging(ec);

        launch_two_devices(ec, executable);

        const std::string replay_label = std::string(label) + " replay " + std::to_string(replay);
        set_device(ec, 0);
        const auto bits_0 = from_device<std::uint16_t>(buffer_0.data(), count);
        failures += verify_pointwise((replay_label + " device 0").c_str(),
                                     from_device_bf16(buffer_0.data(), count), expected,
                                     allreduce_sum_bf16_criterion());
        set_device(ec, 1);
        failures += verify_exact((replay_label + " device 1 equals device 0").c_str(),
                                 from_device<std::uint16_t>(buffer_1.data(), count), bits_0);
    }

    // Back-to-back launches: every site doubles the exact power-of-two multiple of bf16(a + b).
    const double back_to_back = std::ldexp(1.0, kBackToBack * sites);
    for (double& value : expected) { value *= back_to_back; }
    set_device(ec, 0);
    for (int launch = 0; launch < kBackToBack; ++launch) { executable.launch(ec.dev[0]->stream); }
    synchronize_both(ec);
    const std::string chained_label = std::string(label) + " back-to-back launches";
    set_device(ec, 0);
    const auto chained_0 = from_device<std::uint16_t>(buffer_0.data(), count);
    failures += verify_pointwise((chained_label + " device 0").c_str(),
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    set_device(ec, 1);
    failures += verify_exact((chained_label + " device 1 equals device 0").c_str(),
                             from_device<std::uint16_t>(buffer_1.data(), count), chained_0);

    // Eager, staged, after the captured history: exactly doubles the last launch's result.
    for (double& value : expected) { value *= 2.0; }
    ops::allreduce_sum(buffer, staging, ec, events);
    synchronize_both(ec);
    const std::string eager_label = std::string(label) + " eager after replays";
    set_device(ec, 0);
    failures += verify_pointwise((eager_label + " device 0").c_str(),
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("captured buffer device 0");
    failures += staging_0.verify_guards("captured staging device 0");
    set_device(ec, 1);
    failures += verify_pointwise((eager_label + " device 1").c_str(),
                                 from_device_bf16(buffer_1.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("captured buffer device 1");
    failures += staging_1.verify_guards("captured staging device 1");
    if (mailbox != nullptr && mailbox->hang_reported()) {
        std::cerr << label << ": a mailbox exchange reported a hang\n";
        ++failures;
    }
    return failures;
}

// Informative replay cost of the decode token's 128 all-reduces of one [5120] row, per transport:
// one graph of kSites all-reduces replayed kReplays times, reported per all-reduce. Inputs are
// zero, so the values stay bounded.
void run_captured_microbenchmark(const char* label, const ExecutionContext& ec,
                                 const ops::PeerEvents& events) {
    constexpr std::int32_t n = 5120;
    constexpr int kSites     = 128;
    constexpr int kWarmup    = 5;
    constexpr int kReplays   = 50;
    const std::size_t bytes  = static_cast<std::size_t>(n) * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.fill(0);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.fill(0);
    staging_1.fill(0);
    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};
    retire_staging(ec);

    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        for (int site = 0; site < kSites; ++site) {
            ops::allreduce_sum(buffer, staging, ec, events);
        }
    });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    for (int i = 0; i < kWarmup; ++i) { launch_two_devices(ec, executable); }

    const auto started = std::chrono::steady_clock::now();
    set_device(ec, 0);
    for (int i = 0; i < kReplays; ++i) { executable.launch(ec.dev[0]->stream); }
    synchronize_both(ec);
    const std::chrono::duration<double, std::micro> elapsed =
        std::chrono::steady_clock::now() - started;
    std::cout << "allreduce captured replay (" << label << "): " << bytes << " B bf16, "
              << elapsed.count() / (static_cast<double>(kReplays) * kSites)
              << " us per all-reduce over " << kReplays << " replays of " << kSites
              << " (informative)\n";
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
        std::cout << "SKIP: cross-device collectives require two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    const ExecutionContext ec({0, 1});
    const bool peer_access = ops::enable_peer_access(ec);
    std::cout << "peer access: "
              << (peer_access ? "enabled (direct P2P)"
                              : "unavailable (CUDA stages the device-to-device copies through "
                                "host memory)")
              << '\n';
    const ops::PeerEvents events(ec);

    int failures = 0;
    // Real decode shape first: 5120 is the hidden dimension all-reduced 128 times per token.
    failures += run_allreduce_case("allreduce_sum [5120]", 5120, 1, 101u, ec, events);
    // The real row-parallel residual: a full 48-token prefill chunk, 2-D.
    failures += run_allreduce_case("allreduce_sum [5120,48]", 5120, 48, 102u, ec, events);
    failures += run_allreduce_case("allreduce_sum [4097]", 4097, 1, 103u, ec, events);
    failures += run_allreduce_case("allreduce_sum [1]", 1, 1, 104u, ec, events);
    failures += run_allreduce_case("allreduce_sum [3]", 3, 1, 105u, ec, events);
    failures += run_allreduce_case("allreduce_sum [17,3]", 17, 3, 106u, ec, events);

    failures += run_allgather_case("allgather_rows [5120,1024]", 512, 512, 5120, 201u, ec, events);
    // Gathered logits: 248320 vocabulary rows for one token, split by vocabulary half.
    failures +=
        run_allgather_case("allgather_rows [1,248320]", 124160, 124160, 1, 202u, ec, events);
    failures += run_allgather_case("allgather_rows [5120,3] uneven", 2, 1, 5120, 203u, ec, events);
    failures += run_allgather_case("allgather_rows [7,2] minimal", 1, 1, 7, 204u, ec, events);

    failures += run_chained_case(ec, events);

    // Captured transports. The mailbox slot holds the MTP-3 verification activation [5120, 4];
    // the [5120, 8] case exceeds it and must stay staged inside the same kind of graph.
    ops::PeerMailbox mailbox(ec, 5120 * 4 * sizeof(std::uint16_t));
    ops::PeerEvents mailbox_events(ec);
    mailbox_events.attach_mailbox(&mailbox);
    failures += run_captured_case("captured staged [5120]", 5120, 1, 5, false, ec, events, nullptr);
    failures +=
        run_captured_case("captured staged [5120,4]", 5120, 4, 5, false, ec, events, nullptr);
    failures += run_captured_case("captured mailbox [5120]", 5120, 1, 5, true, ec, mailbox_events,
                                  &mailbox);
    failures += run_captured_case("captured mailbox [5120,4]", 5120, 4, 5, true, ec, mailbox_events,
                                  &mailbox);
    failures += run_captured_case("captured mailbox [8,1] minimal", 8, 1, 3, true, ec,
                                  mailbox_events, &mailbox);
    failures += run_captured_case("captured oversized [5120,8] stays staged", 5120, 8, 3, false, ec,
                                  mailbox_events, &mailbox);
    // Unaligned vector count: 5121 elements are not whole 16-byte vectors.
    failures += run_captured_case("captured [5121] stays staged", 5121, 1, 3, false, ec,
                                  mailbox_events, &mailbox);

    failures += run_microbenchmark(ec, events);
    run_captured_microbenchmark("staged", ec, events);
    run_captured_microbenchmark("mailbox", ec, mailbox_events);

    std::cout << (failures ? "FAIL" : "OK") << " allreduce\n";
    return failures ? 1 : 0;
}
