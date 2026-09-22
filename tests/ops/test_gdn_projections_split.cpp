// Two-device parity of the GDN projection split forms: gdn_input_proj_column_parallel and its
// convolution snapshot/record forms (FP8), and gdn_gating_proj_column_parallel (BF16).
//
// Each case runs the single-device Op on device 0 over the whole parent and the split form over
// the two shards, then compares every section a rank owns with the matching rows of the
// single-device result. The GDN input parent [16384,5120] is Q[0,2048) | K[2048,4096) |
// V[4096,10240) | Z[10240,16384); rank r's shard concatenates its Q, K, V and Z rows in that order,
// gathered from the parent's stored rows as a two-device loader would, and its convolution
// channels are the matching Q|K|V channels. The gating shard holds value heads [24r,24r+24).
//
// Every shard weight is prepared through ops::prepare_*_weights from a per-rank parent, so the
// suite also covers the shard geometries the native preparation accepts.
//
// Column-parallel ranks evaluate the dot products the whole-parent kernel evaluates for their
// rows, so the result is expected to match to rounding: two BF16 ulp of the largest output and of
// relative L2. The single-device FP8 snapshot and record may use fused kernels that keep the new
// projection private, while the shard composes it through a BF16 plane, which is the same bound.
//
// Every device case needs two CUDA devices in one process and the suite reports 77 with fewer.
// The registry probe is host-only and runs first.
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/weight_input.h"

#include "core/device.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw   = ninfer::test::quantized_weight;
namespace bf16 = ninfer::test::direct_bf16_weight;

namespace {

constexpr double kBf16Ulp = 1.0 / 256.0;
constexpr ReductionCriterion kSplitCriterion{2.0 * kBf16Ulp, 0.0, 2.0 * kBf16Ulp};

constexpr QType kFp8 = QType::FP8_E4M3FN_ROW_BF16;

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kKeyRows    = 2048;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kChannels   = 2 * kKeyRows + kValueRows;
constexpr std::int32_t kParentRows = kChannels + kValueRows;

constexpr std::int32_t kShardKeyRows   = kKeyRows / 2;
constexpr std::int32_t kShardValueRows = kValueRows / 2;
constexpr std::int32_t kShardChannels  = 2 * kShardKeyRows + kShardValueRows;
constexpr std::int32_t kShardRows      = kShardChannels + kShardValueRows;

constexpr std::int32_t kHeads      = 48;
constexpr std::int32_t kShardHeads = 24;

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

// Parent row of rank `rank`'s shard-local row. Rows below kShardChannels are also the shard's
// convolution channels, mapped to the parent's.
std::int32_t parent_row(std::int32_t local, int rank) {
    if (local < kShardKeyRows) { return rank * kShardKeyRows + local; }
    if (local < 2 * kShardKeyRows) {
        return kKeyRows + rank * kShardKeyRows + local - kShardKeyRows;
    }
    if (local < kShardChannels) {
        return 2 * kKeyRows + rank * kShardValueRows + local - 2 * kShardKeyRows;
    }
    return kChannels + rank * kShardValueRows + local - kShardChannels;
}

// parent_row both builds and checks the shards, so it is pinned here to the literal section
// starts and checked to partition the parent between the two ranks.
int verify_section_map() {
    std::vector<int> owner(kParentRows, -1);
    for (int rank = 0; rank < 2; ++rank) {
        const std::array<std::int32_t, 4> starts{rank * 1024, 2048 + rank * 1024,
                                                 4096 + rank * 3072, 10240 + rank * 3072};
        const std::array<std::int32_t, 4> locals{0, 1024, 2048, 5120};
        for (std::size_t section = 0; section < starts.size(); ++section) {
            if (parent_row(locals[section], rank) != starts[section]) {
                std::cerr << "section map: rank " << rank << " section " << section
                          << " starts at the wrong parent row\n";
                return 1;
            }
        }
        for (std::int32_t local = 0; local < kShardRows; ++local) {
            const std::int32_t row = parent_row(local, rank);
            if (row < 0 || row >= kParentRows || owner[static_cast<std::size_t>(row)] != -1) {
                std::cerr << "section map: parent row " << row << " is claimed twice\n";
                return 1;
            }
            owner[static_cast<std::size_t>(row)] = rank;
        }
    }
    if (std::find(owner.begin(), owner.end(), -1) != owner.end()) {
        std::cerr << "section map: a parent row is unowned\n";
        return 1;
    }
    return 0;
}

// Rank `rank`'s standalone FP8 shard, gathered from the parent's stored code rows and row scales.
qw::PackedWeight gather_shard(const qw::PackedWeight& parent, int rank) {
    const auto k = static_cast<std::size_t>(parent.weight.k);
    qw::PackedWeight shard;
    shard.code_plane_bytes   = static_cast<std::uint64_t>(kShardRows) * k;
    shard.scale_plane_offset = (shard.code_plane_bytes + 255U) & ~std::uint64_t{255};
    shard.scale_plane_bytes  = static_cast<std::uint64_t>(kShardRows) * 2;
    shard.payload.assign(
        static_cast<std::size_t>(shard.scale_plane_offset + shard.scale_plane_bytes), 0);
    for (std::int32_t local = 0; local < kShardRows; ++local) {
        const auto source = static_cast<std::size_t>(parent_row(local, rank));
        const auto target = static_cast<std::size_t>(local);
        std::memcpy(shard.payload.data() + target * k, parent.payload.data() + source * k, k);
        std::memcpy(shard.payload.data() + shard.scale_plane_offset + target * 2,
                    parent.payload.data() + parent.scale_plane_offset + source * 2, 2);
    }
    shard.weight                 = parent.weight;
    shard.weight.n               = kShardRows;
    shard.weight.shape[0]        = kShardRows;
    shard.weight.padded_shape[0] = kShardRows;
    shard.weight.scale_ne[0]     = kShardRows;
    shard.weight.scale_nb[1]     = static_cast<std::int64_t>(kShardRows) * 2;
    shard.weight.scale_nb[2]     = shard.weight.scale_nb[1];
    shard.weight.scale_nb[3]     = shard.weight.scale_nb[1];
    shard.weight.payload         = shard.payload.data();
    shard.weight.payload_bytes   = shard.payload.size();
    shard.weight.qdata           = shard.payload.data();
    shard.weight.scales          = shard.payload.data() + shard.scale_plane_offset;
    return shard;
}

int verify_shard_rows(const std::string& label, const qw::PackedWeight& parent,
                      const qw::PackedWeight& shard, int rank) {
    for (const std::int32_t local : {0, 1, 1023, 1024, 2047, 2048, 5119, 5120, 8191}) {
        for (const std::int32_t column : {0, 1, 127, 2560, kHidden - 1}) {
            if (qw::logical_weight_fp64(shard, local, column) !=
                qw::logical_weight_fp64(parent, parent_row(local, rank), column)) {
                std::cerr << label << ": shard row " << local << " is not parent row "
                          << parent_row(local, rank) << '\n';
                return 1;
            }
        }
    }
    return 0;
}

template <typename Bytes>
int verify_distinct(const std::string& label, const Bytes& first, const Bytes& second) {
    if (first != second) { return 0; }
    std::cerr << label << ": the two ranks' data are byte-identical\n";
    return 1;
}

int compare(const std::string& label, const std::vector<double>& got,
            const std::vector<double>& expected) {
    const ReductionStats stats =
        compute_reduction_stats(got.data(), expected.data(), static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2
              << " gross_limit=" << gross_error_limit(stats, kSplitCriterion) << '\n';
    return verify_reduction(label, got, expected, kSplitCriterion);
}

// Rows [row_offset, row_offset + rows) of every column of a column-major [stride, columns] block.
std::vector<double> extract_rows(const std::vector<double>& buffer, std::int32_t stride,
                                 std::int32_t row_offset, std::int32_t rows, std::int32_t columns) {
    std::vector<double> block(static_cast<std::size_t>(rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const auto source = static_cast<std::size_t>(column) * stride + row_offset;
        std::copy_n(buffer.begin() + static_cast<std::ptrdiff_t>(source), rows,
                    block.begin() + static_cast<std::ptrdiff_t>(column) * rows);
    }
    return block;
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

bool same_native_weight(const Weight& lhs, const Weight& rhs) {
    return lhs.qtype == rhs.qtype && lhs.layout == rhs.layout && lhs.n == rhs.n && lhs.k == rhs.k &&
           lhs.qdata == rhs.qdata && lhs.scales == rhs.scales && lhs.payload == rhs.payload &&
           lhs.payload_bytes >= rhs.payload_bytes;
}

// The shard as native preparation binds it: one contiguous per-rank parent whose four logical
// sections are Q, K, V and Z.
Weight prepare_input_shard(const Weight& stored) {
    const std::array<std::uint64_t, 2> shape{kShardRows, kHidden};
    const WeightParent parent{weight_geometry(kFp8, QuantLayout::RowScale, shape),
                              static_cast<const std::byte*>(stored.payload)};
    const auto rows = [&](std::uint64_t begin, std::uint64_t count) {
        return WeightView{{count, kHidden},
                          {{&parent, begin * kHidden, (begin + count) * kHidden}}};
    };
    const WeightView q  = rows(0, kShardKeyRows);
    const WeightView k  = rows(kShardKeyRows, kShardKeyRows);
    const WeightView v  = rows(2 * kShardKeyRows, kShardValueRows);
    const WeightView z  = rows(kShardChannels, kShardValueRows);
    const auto prepared = ops::prepare_gdn_input_proj_weights({q}, {k}, {v}, {z});
    return std::get<ops::SingleProjectionWeight>(prepared).weight;
}

// Parent, shards and their device copies of the GDN input weight.
struct InputWeights {
    RankWeight parent;
    std::array<RankWeight, 2> shard;
    std::array<Weight, 2> prepared{};
};

int make_input_weights(const ExecutionContext& ec, std::uint32_t seed, InputWeights& out) {
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    const qw::PackedWeight parent =
        qw::make_patterned_weight(kFp8, kParentRows, kHidden, seed, options);
    const std::array<qw::PackedWeight, 2> shard{gather_shard(parent, 0), gather_shard(parent, 1)};
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        failures += verify_shard_rows("fp8 gdn_input shard " + std::to_string(rank), parent,
                                      shard[static_cast<std::size_t>(rank)], rank);
    }
    failures += verify_distinct("fp8 gdn_input shards", shard[0].payload, shard[1].payload);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    out.parent = upload(parent);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        out.shard[slot]    = upload(shard[slot]);
        out.prepared[slot] = prepare_input_shard(out.shard[slot].weight);
        if (!same_native_weight(out.prepared[slot], out.shard[slot].weight)) {
            std::cerr << "prepare_gdn_input_proj_weights: rank " << rank
                      << " shard does not bind the stored parent\n";
            ++failures;
        }
    }
    return failures;
}

std::vector<float> activation(std::size_t elements, std::uint32_t seed) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, -1.0F, 1.0F);
    round_to_bf16(values);
    return values;
}

// ================================================================================================
// gdn_input_proj_column_parallel
// ================================================================================================

int run_input_case(const ExecutionContext& ec, const InputWeights& weights, std::uint32_t seed) {
    std::cout << "fp8 gdn_input_proj [" << kParentRows << ',' << kHidden << "] -> [" << kShardRows
              << ',' << kHidden << "]\n";
    int failures = 0;
    // Decode, the SIMT and bounded-MMA frontiers, the A8 threshold and the GEMM schedules.
    const std::vector<std::int32_t> tokens_sweep{1,  2,  3,  4,  5,  7,  8,   9,
                                                 16, 17, 33, 64, 65, 97, 128, 300};
    const std::vector<ops::LinearPolicy> policies{
        ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4};

    for (const std::int32_t tokens : tokens_sweep) {
        const std::vector<float> x_host =
            activation(static_cast<std::size_t>(kHidden) * tokens,
                       seed * 31U + static_cast<std::uint32_t>(tokens));
        set_device(ec, 0);
        const DeviceBuffer parent_x = to_device_bf16(x_host);
        std::array<DeviceBuffer, 2> shard_x;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(x_host);
        }

        for (const ops::LinearPolicy policy : policies) {
            const std::string label =
                "fp8 gdn_input T=" + std::to_string(tokens) + " " + policy_name(policy);

            set_device(ec, 0);
            GuardedDeviceBuffer ref_qkv(static_cast<std::size_t>(kChannels) * tokens * 2);
            GuardedDeviceBuffer ref_z(static_cast<std::size_t>(kValueRows) * tokens * 2);
            ref_qkv.fill(0xff);
            ref_z.fill(0xff);
            DeviceArena reference_arena(
                std::max<std::size_t>(ops::gdn_input_proj_workspace_capacity_bytes(
                                          kFp8, kParentRows, kHidden, policy, tokens, tokens),
                                      1));
            Tensor reference_qkv(ref_qkv.data(), DType::BF16, {kChannels, tokens});
            Tensor reference_z(ref_z.data(), DType::BF16, {kValueRows, tokens});
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::gdn_input_proj(Tensor(parent_x.p, DType::BF16, {kHidden, tokens}),
                                weights.parent.weight, reference_qkv, reference_z, policy,
                                reference_arena, ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += ref_qkv.verify_guards(label + " reference qkv");
            failures += ref_z.verify_guards(label + " reference z");
            const std::vector<double> expected_qkv =
                from_device_bf16(ref_qkv.data(), static_cast<std::size_t>(kChannels) * tokens);
            const std::vector<double> expected_z =
                from_device_bf16(ref_z.data(), static_cast<std::size_t>(kValueRows) * tokens);

            const std::size_t capacity =
                ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
                    kFp8, kShardRows, kHidden, policy, tokens, tokens);
            std::array<std::optional<GuardedDeviceBuffer>, 2> qkv_buffer;
            std::array<std::optional<GuardedDeviceBuffer>, 2> z_buffer;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                qkv_buffer[slot].emplace(static_cast<std::size_t>(kShardChannels) * tokens * 2);
                z_buffer[slot].emplace(static_cast<std::size_t>(kShardValueRows) * tokens * 2);
                qkv_buffer[slot]->fill(0xff);
                z_buffer[slot]->fill(0xff);
                arena[slot].emplace(std::max<std::size_t>(capacity, 1));
            }
            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kHidden, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {kHidden, tokens})};
            const std::array<Tensor, 2> qkv{
                Tensor(qkv_buffer[0]->data(), DType::BF16, {kShardChannels, tokens}),
                Tensor(qkv_buffer[1]->data(), DType::BF16, {kShardChannels, tokens})};
            const std::array<Tensor, 2> z{
                Tensor(z_buffer[0]->data(), DType::BF16, {kShardValueRows, tokens}),
                Tensor(z_buffer[1]->data(), DType::BF16, {kShardValueRows, tokens})};
            const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            if (policy == ops::LinearPolicy::A16Only) {
                ops::gdn_input_proj_column_parallel(x, weights.prepared, qkv, z, ec);
            } else {
                ops::gdn_input_proj_column_parallel(x, weights.prepared, qkv, z, policy, workspace,
                                                    ec);
            }
            synchronize_both(ec);

            std::array<std::vector<double>, 2> observed_qkv;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot          = static_cast<std::size_t>(rank);
                const std::string prefix = label + " rank " + std::to_string(rank);
                set_device(ec, rank);
                failures += qkv_buffer[slot]->verify_guards(prefix + " qkv");
                failures += z_buffer[slot]->verify_guards(prefix + " z");
                observed_qkv[slot] = from_device_bf16(
                    qkv_buffer[slot]->data(), static_cast<std::size_t>(kShardChannels) * tokens);
                const std::vector<double> observed_z = from_device_bf16(
                    z_buffer[slot]->data(), static_cast<std::size_t>(kShardValueRows) * tokens);
                failures += compare(
                    prefix + " q",
                    extract_rows(observed_qkv[slot], kShardChannels, 0, kShardKeyRows, tokens),
                    extract_rows(expected_qkv, kChannels, parent_row(0, rank), kShardKeyRows,
                                 tokens));
                failures +=
                    compare(prefix + " k",
                            extract_rows(observed_qkv[slot], kShardChannels, kShardKeyRows,
                                         kShardKeyRows, tokens),
                            extract_rows(expected_qkv, kChannels, parent_row(kShardKeyRows, rank),
                                         kShardKeyRows, tokens));
                failures += compare(prefix + " v",
                                    extract_rows(observed_qkv[slot], kShardChannels,
                                                 2 * kShardKeyRows, kShardValueRows, tokens),
                                    extract_rows(expected_qkv, kChannels,
                                                 parent_row(2 * kShardKeyRows, rank),
                                                 kShardValueRows, tokens));
                failures += compare(prefix + " z", observed_z,
                                    extract_rows(expected_z, kValueRows,
                                                 parent_row(kShardChannels, rank) - kChannels,
                                                 kShardValueRows, tokens));
            }
            if (observed_qkv[0] == observed_qkv[1]) {
                std::cerr << label << ": both ranks wrote the same qkv\n";
                ++failures;
            }
        }
    }
    return failures;
}

// ================================================================================================
// gdn_input_proj_conv_snapshot_column_parallel / gdn_input_proj_conv_record_column_parallel
// ================================================================================================

struct ConvCase {
    std::int32_t batch;
    std::int32_t width;
    bool mixed; // Mixed valid-column extents.
};

// Host data of one convolution case at the parent geometry. BF16 values are held as bits.
struct ConvData {
    std::int32_t batch = 0;
    std::int32_t width = 0;
    std::int32_t slots = 0;
    std::vector<std::uint16_t> x;
    std::vector<std::uint16_t> conv_weight; // [channels, 4]
    std::vector<std::uint16_t> states;      // [channels, 3, slots]
    std::vector<std::int32_t> valid;        // empty or [batch]
    std::vector<std::int32_t> initial;      // [batch]
    std::vector<std::int32_t> base;         // [batch]
};

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

ConvData make_conv_data(const ConvCase& test_case, std::uint32_t seed) {
    ConvData data;
    data.batch = test_case.batch;
    data.width = test_case.width;
    // Row b's history comes from slot B*W+b and its snapshots go to [b*W, b*W+W).
    data.slots = test_case.batch * test_case.width + test_case.batch;
    const auto x =
        activation(static_cast<std::size_t>(kHidden) * test_case.width * test_case.batch, seed);
    data.x = bf16_bits(x);
    std::vector<float> conv(static_cast<std::size_t>(kChannels) * 4);
    fill_uniform(conv, seed + 1, -0.6F, 0.6F);
    data.conv_weight = bf16_bits(conv);
    std::vector<float> states(static_cast<std::size_t>(kChannels) * 3 * data.slots);
    fill_uniform(states, seed + 2, -1.0F, 1.0F);
    data.states = bf16_bits(states);
    for (std::int32_t row = 0; row < test_case.batch; ++row) {
        data.initial.push_back(test_case.batch * test_case.width + row);
        data.base.push_back(row * test_case.width);
        if (test_case.mixed) { data.valid.push_back(1 + (row * 5 + 2) % test_case.width); }
    }
    return data;
}

// Rank `rank`'s channels of a [channels, inner] BF16 block.
std::vector<std::uint16_t> gather_channels(const std::vector<std::uint16_t>& parent,
                                           std::int32_t inner, int rank) {
    std::vector<std::uint16_t> shard(static_cast<std::size_t>(kShardChannels) * inner);
    for (std::int32_t index = 0; index < inner; ++index) {
        for (std::int32_t local = 0; local < kShardChannels; ++local) {
            shard[static_cast<std::size_t>(index) * kShardChannels + local] =
                parent[static_cast<std::size_t>(index) * kChannels + parent_row(local, rank)];
        }
    }
    return shard;
}

std::vector<double> promote(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { values[i] = bf16_to_f32(bits[i]); }
    return values;
}

// Device operands of one convolution call, allocated with the owning device current.
struct ConvOperands {
    DeviceBuffer x, conv_weight, states, valid, initial, base, record;
    std::optional<GuardedDeviceBuffer> query, key, value, z;

    ConvOperands(const ConvData& data, const std::vector<std::uint16_t>& conv_weight_bits,
                 const std::vector<std::uint16_t>& state_bits, std::int32_t channels,
                 std::int32_t key_rows, std::int32_t value_rows) {
        const auto columns = static_cast<std::size_t>(data.width) * data.batch;
        x                  = to_device(data.x);
        conv_weight        = to_device(conv_weight_bits);
        states             = to_device(state_bits);
        if (!data.valid.empty()) { valid = to_device(data.valid); }
        initial = to_device(data.initial);
        base    = to_device(data.base);
        record  = DeviceBuffer(static_cast<std::size_t>(channels) * columns * 2);
        query.emplace(static_cast<std::size_t>(key_rows) * columns * 2);
        key.emplace(static_cast<std::size_t>(key_rows) * columns * 2);
        value.emplace(static_cast<std::size_t>(value_rows) * columns * 2);
        z.emplace(static_cast<std::size_t>(value_rows) * columns * 2);
        for (auto* buffer : {&*query, &*key, &*value, &*z}) { buffer->fill(0xff); }
    }
};

struct ConvTensors {
    Tensor x, conv_weight, states, valid, initial, base, record, query, key, value, z;
};

ConvTensors conv_tensors(ConvOperands& operands, const ConvData& data, std::int32_t hidden,
                         std::int32_t channels, std::int32_t key_rows, std::int32_t value_rows) {
    const std::int32_t w = data.width;
    const std::int32_t b = data.batch;
    ConvTensors t;
    t.x           = Tensor(operands.x.p, DType::BF16, {hidden, w, b});
    t.conv_weight = Tensor(operands.conv_weight.p, DType::BF16, {channels, 4});
    t.states      = Tensor(operands.states.p, DType::BF16, {channels, 3, data.slots});
    if (!data.valid.empty()) { t.valid = Tensor(operands.valid.p, DType::I32, {b}); }
    t.initial = Tensor(operands.initial.p, DType::I32, {b});
    t.base    = Tensor(operands.base.p, DType::I32, {b});
    t.record  = Tensor(operands.record.p, DType::BF16, {channels, w, b});
    t.query   = Tensor(operands.query->data(), DType::BF16, {key_rows, w, b});
    t.key     = Tensor(operands.key->data(), DType::BF16, {key_rows, w, b});
    t.value   = Tensor(operands.value->data(), DType::BF16, {value_rows, w, b});
    t.z       = Tensor(operands.z->data(), DType::BF16, {value_rows, w, b});
    return t;
}

struct ConvResult {
    std::vector<double> query, key, value, z, record;
    std::vector<std::uint16_t> states;
};

ConvResult read_conv(const ConvOperands& operands, const ConvData& data, std::int32_t channels,
                     std::int32_t key_rows, std::int32_t value_rows, bool record) {
    const auto columns = static_cast<std::size_t>(data.width) * data.batch;
    ConvResult result;
    result.query  = from_device_bf16(operands.query->data(), key_rows * columns);
    result.key    = from_device_bf16(operands.key->data(), key_rows * columns);
    result.value  = from_device_bf16(operands.value->data(), value_rows * columns);
    result.z      = from_device_bf16(operands.z->data(), value_rows * columns);
    result.states = from_device<std::uint16_t>(operands.states,
                                               static_cast<std::size_t>(channels) * 3 * data.slots);
    if (record) { result.record = from_device_bf16(operands.record.p, channels * columns); }
    return result;
}

int verify_conv_guards(const ConvOperands& operands, const std::string& label) {
    return operands.query->verify_guards(label + " query") +
           operands.key->verify_guards(label + " key") +
           operands.value->verify_guards(label + " value") +
           operands.z->verify_guards(label + " z");
}

int compare_conv_rank(const std::string& label, const ConvResult& shard, const ConvResult& parent,
                      const ConvData& data, int rank, bool record) {
    const std::int32_t columns = data.width * data.batch;
    int failures               = 0;
    failures +=
        compare(label + " query", shard.query,
                extract_rows(parent.query, kKeyRows, parent_row(0, rank), kShardKeyRows, columns));
    failures +=
        compare(label + " key", shard.key,
                extract_rows(parent.key, kKeyRows, parent_row(kShardKeyRows, rank) - kKeyRows,
                             kShardKeyRows, columns));
    failures += compare(label + " value", shard.value,
                        extract_rows(parent.value, kValueRows,
                                     parent_row(2 * kShardKeyRows, rank) - 2 * kKeyRows,
                                     kShardValueRows, columns));
    failures +=
        compare(label + " z", shard.z,
                extract_rows(parent.z, kValueRows, parent_row(kShardChannels, rank) - kChannels,
                             kShardValueRows, columns));
    // State slots and record columns are channel-major blocks; gather the rank's channels.
    failures += compare(label + " states", promote(shard.states),
                        promote(gather_channels(parent.states, 3 * data.slots, rank)));
    if (record) {
        std::vector<std::uint16_t> parent_record(parent.record.size());
        for (std::size_t i = 0; i < parent_record.size(); ++i) {
            parent_record[i] = f32_to_bf16(static_cast<float>(parent.record[i]));
        }
        std::vector<double> expected = promote(gather_channels(parent_record, columns, rank));
        std::vector<double> got      = shard.record;
        // Only the valid prefix of each row's record is defined.
        if (!data.valid.empty()) {
            for (std::int32_t row = 0; row < data.batch; ++row) {
                for (std::int32_t column = data.valid[static_cast<std::size_t>(row)];
                     column < data.width; ++column) {
                    const auto offset =
                        static_cast<std::size_t>(row * data.width + column) * kShardChannels;
                    std::fill_n(expected.begin() + static_cast<std::ptrdiff_t>(offset),
                                kShardChannels, 0.0);
                    std::fill_n(got.begin() + static_cast<std::ptrdiff_t>(offset), kShardChannels,
                                0.0);
                }
            }
        }
        failures += compare(label + " record", got, expected);
    }
    return failures;
}

int run_conv_case(const ExecutionContext& ec, const InputWeights& weights,
                  const ConvCase& test_case, ops::LinearPolicy policy, bool record,
                  std::uint32_t seed) {
    const std::string label =
        std::string("fp8 gdn_input_proj_conv_") + (record ? "record" : "snapshot") +
        " B=" + std::to_string(test_case.batch) + " W=" + std::to_string(test_case.width) +
        (test_case.mixed ? " mixed " : " ") + policy_name(policy);
    const ConvData data = make_conv_data(test_case, seed);
    int failures        = 0;

    // Single-device reference over the whole parent.
    set_device(ec, 0);
    ConvOperands parent_operands(data, data.conv_weight, data.states, kChannels, kKeyRows,
                                 kValueRows);
    ConvTensors parent =
        conv_tensors(parent_operands, data, kHidden, kChannels, kKeyRows, kValueRows);
    {
        const std::size_t capacity =
            record ? ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                         kFp8, kParentRows, kHidden, policy, data.batch, data.width, data.width)
                   : ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                         kFp8, kParentRows, kHidden, policy, data.batch, data.width, data.width);
        DeviceArena arena(std::max<std::size_t>(capacity, 1));
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (record) {
            ops::gdn_input_proj_conv_record(parent.x, weights.parent.weight, parent.conv_weight,
                                            parent.states, parent.valid, parent.initial,
                                            parent.record, parent.query, parent.key, parent.value,
                                            parent.z, policy, arena, ec.dev[0]->stream);
        } else {
            ops::gdn_input_proj_conv_snapshot(parent.x, weights.parent.weight, parent.conv_weight,
                                              parent.states, parent.valid, parent.initial,
                                              parent.base, parent.query, parent.key, parent.value,
                                              parent.z, policy, arena, ec.dev[0]->stream);
        }
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
    }
    failures += verify_conv_guards(parent_operands, label + " reference");
    const ConvResult expected =
        read_conv(parent_operands, data, kChannels, kKeyRows, kValueRows, record);

    // Split form over the two shards.
    const std::size_t capacity =
        record ? ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
                     kFp8, kShardRows, kHidden, policy, data.batch, data.width, data.width)
               : ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
                     kFp8, kShardRows, kHidden, policy, data.batch, data.width, data.width);
    std::array<std::optional<ConvOperands>, 2> operands;
    std::array<std::optional<DeviceArena>, 2> arena;
    std::array<ConvTensors, 2> t;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        operands[slot].emplace(data, gather_channels(data.conv_weight, 4, rank),
                               gather_channels(data.states, 3 * data.slots, rank), kShardChannels,
                               kShardKeyRows, kShardValueRows);
        arena[slot].emplace(std::max<std::size_t>(capacity, 1));
        t[slot] = conv_tensors(*operands[slot], data, kHidden, kShardChannels, kShardKeyRows,
                               kShardValueRows);
    }
    const auto pick = [&](Tensor ConvTensors::* member) {
        return std::array<Tensor, 2>{t[0].*member, t[1].*member};
    };
    const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};
    retire_staging(ec);
    if (record) {
        ops::gdn_input_proj_conv_record_column_parallel(
            pick(&ConvTensors::x), weights.prepared, pick(&ConvTensors::conv_weight),
            pick(&ConvTensors::states), pick(&ConvTensors::valid), pick(&ConvTensors::initial),
            pick(&ConvTensors::record), pick(&ConvTensors::query), pick(&ConvTensors::key),
            pick(&ConvTensors::value), pick(&ConvTensors::z), policy, workspace, ec);
    } else {
        ops::gdn_input_proj_conv_snapshot_column_parallel(
            pick(&ConvTensors::x), weights.prepared, pick(&ConvTensors::conv_weight),
            pick(&ConvTensors::states), pick(&ConvTensors::valid), pick(&ConvTensors::initial),
            pick(&ConvTensors::base), pick(&ConvTensors::query), pick(&ConvTensors::key),
            pick(&ConvTensors::value), pick(&ConvTensors::z), policy, workspace, ec);
    }
    synchronize_both(ec);

    std::array<ConvResult, 2> observed;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot          = static_cast<std::size_t>(rank);
        const std::string prefix = label + " rank " + std::to_string(rank);
        set_device(ec, rank);
        failures += verify_conv_guards(*operands[slot], prefix);
        observed[slot] = read_conv(*operands[slot], data, kShardChannels, kShardKeyRows,
                                   kShardValueRows, record);
        failures += compare_conv_rank(prefix, observed[slot], expected, data, rank, record);
    }
    if (observed[0].value == observed[1].value) {
        std::cerr << label << ": both ranks wrote the same value output\n";
        ++failures;
    }

    // The record form's outputs equal the snapshot form's from the same history. The record
    // left the initial slots untouched, so the snapshot can run on the same operands.
    if (record) {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            for (auto* buffer : {&*operands[static_cast<std::size_t>(rank)]->query,
                                 &*operands[static_cast<std::size_t>(rank)]->key,
                                 &*operands[static_cast<std::size_t>(rank)]->value,
                                 &*operands[static_cast<std::size_t>(rank)]->z}) {
                buffer->fill(0xff);
            }
        }
        const std::size_t snapshot_capacity =
            ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
                kFp8, kShardRows, kHidden, policy, data.batch, data.width, data.width);
        std::array<std::optional<DeviceArena>, 2> snapshot_arena;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            snapshot_arena[static_cast<std::size_t>(rank)].emplace(
                std::max<std::size_t>(snapshot_capacity, 1));
        }
        const std::array<WorkspaceArena*, 2> snapshot_workspace{&*snapshot_arena[0],
                                                                &*snapshot_arena[1]};
        retire_staging(ec);
        ops::gdn_input_proj_conv_snapshot_column_parallel(
            pick(&ConvTensors::x), weights.prepared, pick(&ConvTensors::conv_weight),
            pick(&ConvTensors::states), pick(&ConvTensors::valid), pick(&ConvTensors::initial),
            pick(&ConvTensors::base), pick(&ConvTensors::query), pick(&ConvTensors::key),
            pick(&ConvTensors::value), pick(&ConvTensors::z), policy, snapshot_workspace, ec);
        synchronize_both(ec);
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            const ConvResult snapshot = read_conv(*operands[slot], data, kShardChannels,
                                                  kShardKeyRows, kShardValueRows, false);
            if (snapshot.query != observed[slot].query || snapshot.key != observed[slot].key ||
                snapshot.value != observed[slot].value || snapshot.z != observed[slot].z) {
                std::cerr << label << " rank " << rank
                          << ": record outputs differ from the snapshot form's\n";
                ++failures;
            }
        }
    }
    return failures;
}

// ================================================================================================
// gdn_gating_proj_column_parallel
// ================================================================================================

bf16::HostWeight rows_of(const bf16::HostWeight& parent, std::int32_t begin, std::int32_t rows) {
    bf16::HostWeight block;
    block.n = rows;
    block.k = parent.k;
    block.bits.assign(parent.bits.begin() + static_cast<std::ptrdiff_t>(begin) * parent.k,
                      parent.bits.begin() + static_cast<std::ptrdiff_t>(begin + rows) * parent.k);
    return block;
}

// The rank's contiguous [48,5120] A|B parent, prepared as the split form's fused weight.
Weight prepare_gating_shard(const void* payload) {
    const std::array<std::uint64_t, 2> shape{2 * kShardHeads, kHidden};
    const WeightParent parent{weight_geometry(QType::BF16, QuantLayout::Contiguous, shape),
                              static_cast<const std::byte*>(payload)};
    const std::uint64_t count = static_cast<std::uint64_t>(kShardHeads) * kHidden;
    const std::vector<std::uint64_t> logical{kShardHeads, kHidden};
    const WeightView a{logical, {{&parent, 0, count}}};
    const WeightView b{logical, {{&parent, count, 2 * count}}};
    return std::get<ops::SingleProjectionWeight>(ops::prepare_gdn_gating_proj_weights({a}, {b}))
        .weight;
}

int run_gating_case(const ExecutionContext& ec, std::uint32_t seed, bool fused) {
    const std::string head = std::string("gdn_gating_proj ") + (fused ? "ab parent" : "a/b");
    std::cout << head << " [" << kHeads << ',' << kHidden << "] -> [" << kShardHeads << ','
              << kHidden << "]\n";
    int failures = 0;

    const bf16::HostWeight a = bf16::make_patterned(kHeads, kHidden, seed);
    const bf16::HostWeight b = bf16::make_patterned(kHeads, kHidden, seed + 1);
    std::vector<float> a_log(kHeads);
    std::vector<float> dt_bias(kHeads);
    fill_uniform(a_log, seed + 2, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 3, -1.0F, 1.0F);

    std::array<bf16::HostWeight, 2> shard_a{rows_of(a, 0, kShardHeads),
                                            rows_of(a, kShardHeads, kShardHeads)};
    std::array<bf16::HostWeight, 2> shard_b{rows_of(b, 0, kShardHeads),
                                            rows_of(b, kShardHeads, kShardHeads)};
    failures += verify_distinct(head + " a", shard_a[0].bits, shard_a[1].bits);
    failures += verify_distinct(head + " b", shard_b[0].bits, shard_b[1].bits);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    const DeviceBuffer parent_a       = to_device(a.bits);
    const DeviceBuffer parent_b       = to_device(b.bits);
    const DeviceBuffer parent_a_log   = to_device_f32(a_log);
    const DeviceBuffer parent_dt_bias = to_device_f32(dt_bias);
    std::array<DeviceBuffer, 2> ab_device, a_device, b_device, a_log_device, dt_bias_device;
    std::array<Weight, 2> a_weight{}, b_weight{}, ab_weight{};
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        if (fused) {
            std::vector<std::uint16_t> ab = shard_a[slot].bits;
            ab.insert(ab.end(), shard_b[slot].bits.begin(), shard_b[slot].bits.end());
            ab_device[slot] = to_device(ab);
            ab_weight[slot] = prepare_gating_shard(ab_device[slot].p);
        } else {
            a_device[slot] = to_device(shard_a[slot].bits);
            b_device[slot] = to_device(shard_b[slot].bits);
            a_weight[slot] = shard_a[slot].device_weight(a_device[slot].p);
            b_weight[slot] = shard_b[slot].device_weight(b_device[slot].p);
        }
        a_log_device[slot]   = to_device_f32(std::vector<float>(
            a_log.begin() + rank * kShardHeads, a_log.begin() + (rank + 1) * kShardHeads));
        dt_bias_device[slot] = to_device_f32(std::vector<float>(
            dt_bias.begin() + rank * kShardHeads, dt_bias.begin() + (rank + 1) * kShardHeads));
    }

    // GEMV at T=1; the reference crosses from small-T into its MMA routes while the shard keeps
    // the split-10 kernel at every T>=2.
    for (const std::int32_t tokens : {1, 2, 8, 9, 48, 1024}) {
        const std::string label = head + " T=" + std::to_string(tokens);
        const std::vector<float> x_host =
            activation(static_cast<std::size_t>(kHidden) * tokens,
                       seed * 41U + static_cast<std::uint32_t>(tokens));

        set_device(ec, 0);
        const DeviceBuffer parent_x = to_device_bf16(x_host);
        GuardedDeviceBuffer ref_g(static_cast<std::size_t>(kHeads) * tokens * sizeof(float));
        GuardedDeviceBuffer ref_beta(static_cast<std::size_t>(kHeads) * tokens * sizeof(float));
        ref_g.fill(0xff);
        ref_beta.fill(0xff);
        DeviceArena reference_arena(std::max<std::size_t>(
            ops::gdn_gating_proj_workspace_capacity_bytes(kHeads, kHidden, tokens, tokens), 1));
        Tensor reference_g(ref_g.data(), DType::FP32, {kHeads, tokens});
        Tensor reference_beta(ref_beta.data(), DType::FP32, {kHeads, tokens});
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::gdn_gating_proj(Tensor(parent_x.p, DType::BF16, {kHidden, tokens}),
                             a.device_weight(parent_a.p), b.device_weight(parent_b.p),
                             Tensor(parent_a_log.p, DType::FP32, {kHeads}),
                             Tensor(parent_dt_bias.p, DType::FP32, {kHeads}), reference_arena,
                             reference_g, reference_beta, ec.dev[0]->execution_view());
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
        failures += ref_g.verify_guards(label + " reference g");
        failures += ref_beta.verify_guards(label + " reference beta");
        const auto expected_g =
            from_device<float>(ref_g.data(), static_cast<std::size_t>(kHeads) * tokens);
        const auto expected_beta =
            from_device<float>(ref_beta.data(), static_cast<std::size_t>(kHeads) * tokens);

        const std::size_t capacity = ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(
            kShardHeads, kHidden, tokens, tokens);
        std::array<DeviceBuffer, 2> x_device;
        std::array<std::optional<GuardedDeviceBuffer>, 2> g_buffer, beta_buffer;
        std::array<std::optional<DeviceArena>, 2> arena;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            x_device[slot] = to_device_bf16(x_host);
            g_buffer[slot].emplace(static_cast<std::size_t>(kShardHeads) * tokens * sizeof(float));
            beta_buffer[slot].emplace(static_cast<std::size_t>(kShardHeads) * tokens *
                                      sizeof(float));
            g_buffer[slot]->fill(0xff);
            beta_buffer[slot]->fill(0xff);
            arena[slot].emplace(std::max<std::size_t>(capacity, 1));
        }
        const auto per_rank = [&](auto make) { return std::array<Tensor, 2>{make(0), make(1)}; };
        const auto x        = per_rank([&](std::size_t slot) {
            return Tensor(x_device[slot].p, DType::BF16, {kHidden, tokens});
        });
        const auto log_a    = per_rank([&](std::size_t slot) {
            return Tensor(a_log_device[slot].p, DType::FP32, {kShardHeads});
        });
        const auto bias     = per_rank([&](std::size_t slot) {
            return Tensor(dt_bias_device[slot].p, DType::FP32, {kShardHeads});
        });
        const auto g        = per_rank([&](std::size_t slot) {
            return Tensor(g_buffer[slot]->data(), DType::FP32, {kShardHeads, tokens});
        });
        const auto beta     = per_rank([&](std::size_t slot) {
            return Tensor(beta_buffer[slot]->data(), DType::FP32, {kShardHeads, tokens});
        });
        const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

        retire_staging(ec);
        if (fused) {
            ops::gdn_gating_proj_column_parallel(x, ab_weight, log_a, bias, workspace, g, beta, ec);
        } else {
            ops::gdn_gating_proj_column_parallel(x, a_weight, b_weight, log_a, bias, workspace, g,
                                                 beta, ec);
        }
        synchronize_both(ec);

        std::array<std::vector<double>, 2> observed_g;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot          = static_cast<std::size_t>(rank);
            const std::string prefix = label + " rank " + std::to_string(rank);
            set_device(ec, rank);
            failures += g_buffer[slot]->verify_guards(prefix + " g");
            failures += beta_buffer[slot]->verify_guards(prefix + " beta");
            const auto got_g = from_device<float>(g_buffer[slot]->data(),
                                                  static_cast<std::size_t>(kShardHeads) * tokens);
            const auto got_beta = from_device<float>(
                beta_buffer[slot]->data(), static_cast<std::size_t>(kShardHeads) * tokens);
            observed_g[slot] = std::vector<double>(got_g.begin(), got_g.end());
            failures +=
                compare(prefix + " g", observed_g[slot],
                        extract_rows(std::vector<double>(expected_g.begin(), expected_g.end()),
                                     kHeads, rank * kShardHeads, kShardHeads, tokens));
            failures += compare(
                prefix + " beta", std::vector<double>(got_beta.begin(), got_beta.end()),
                extract_rows(std::vector<double>(expected_beta.begin(), expected_beta.end()),
                             kHeads, rank * kShardHeads, kShardHeads, tokens));
        }
        if (observed_g[0] == observed_g[1]) {
            std::cerr << label << ": both ranks wrote the same g\n";
            ++failures;
        }
    }
    return failures;
}

// ================================================================================================
// Host-only registry probe and pair rejections
// ================================================================================================

template <class Body>
int expect_invalid(const std::string& what, Body&& body) {
    try {
        body();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << what << ": wrong exception: " << error.what() << '\n';
        return 1;
    }
    std::cerr << what << ": accepted\n";
    return 1;
}

template <class Body>
int expect_accepted(const std::string& what, Body&& body) {
    try {
        body();
    } catch (const std::exception& error) {
        std::cerr << what << ": rejected: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

int verify_registry() {
    int failures = verify_section_map();
    for (const ops::LinearPolicy policy :
         {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
        for (const std::int32_t tokens : {1, 7, 8, 1024}) {
            failures += expect_accepted("gdn_input workspace", [&] {
                const std::size_t bytes =
                    ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
                        kFp8, kShardRows, kHidden, policy, tokens, tokens);
                // The route frontier is the parent's: A8 from T=8 when the policy permits it.
                const bool a8 = policy != ops::LinearPolicy::A16Only && tokens >= 8;
                if ((bytes != 0) != a8) { throw std::runtime_error("unexpected A8 workspace"); }
            });
        }
        failures += expect_accepted("gdn snapshot workspace", [&] {
            (void)ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
                kFp8, kShardRows, kHidden, policy, 1, 1, 64);
            (void)ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
                kFp8, kShardRows, kHidden, policy, 8, 1, 16);
        });
        failures += expect_accepted("gdn record workspace", [&] {
            (void)ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
                kFp8, kShardRows, kHidden, policy, 8, 2, 16);
        });
    }
    const auto a16 = ops::LinearPolicy::A16Only;
    failures += expect_invalid("gdn_input parent profile", [&] {
        (void)ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(kFp8, kParentRows,
                                                                           kHidden, a16, 1, 1);
    });
    failures += expect_invalid("gdn_input NVFP4 shard", [&] {
        (void)ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(QType::NVFP4, kShardRows,
                                                                           kHidden, a16, 1, 1);
    });
    failures += expect_invalid("gdn_input inverted interval", [&] {
        (void)ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(kFp8, kShardRows,
                                                                           kHidden, a16, 4, 1);
    });
    failures += expect_invalid("gdn snapshot B=2 W=17", [&] {
        (void)ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
            kFp8, kShardRows, kHidden, a16, 2, 1, 17);
    });
    failures += expect_invalid("gdn record W=1", [&] {
        (void)ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
            kFp8, kShardRows, kHidden, a16, 1, 1, 4);
    });
    for (const std::int32_t tokens : {1, 2, 48, 1024}) {
        failures += expect_accepted("gdn_gating workspace", [&] {
            (void)ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(
                kShardHeads, kHidden, tokens, tokens);
        });
    }
    failures += expect_invalid("gdn_gating parent profile", [&] {
        (void)ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(kHeads, kHidden, 1, 1);
    });
    failures += expect_invalid("gdn_gating inverted interval", [&] {
        (void)ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(kShardHeads, kHidden, 4,
                                                                            1);
    });
    std::cout << (failures ? "FAIL" : "OK") << " registry\n";
    return failures;
}

int verify_split_rejections(const ExecutionContext& ec, const InputWeights& weights) {
    int failures = 0;
    std::array<DeviceBuffer, 2> x_buffer, qkv_buffer, z_buffer;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        x_buffer[slot]   = DeviceBuffer(static_cast<std::size_t>(kHidden) * 8 * 2);
        qkv_buffer[slot] = DeviceBuffer(static_cast<std::size_t>(kShardChannels) * 8 * 2);
        z_buffer[slot]   = DeviceBuffer(static_cast<std::size_t>(kShardValueRows) * 8 * 2);
    }
    const auto tensors = [&](std::int32_t first, std::int32_t second, DeviceBuffer* buffers,
                             std::int32_t rows) {
        return std::array<Tensor, 2>{Tensor(buffers[0].p, DType::BF16, {rows, first}),
                                     Tensor(buffers[1].p, DType::BF16, {rows, second})};
    };
    const auto x   = tensors(2, 2, x_buffer.data(), kHidden);
    const auto qkv = tensors(2, 2, qkv_buffer.data(), kShardChannels);
    const auto z   = tensors(2, 2, z_buffer.data(), kShardValueRows);

    failures += expect_invalid("token count", [&] {
        ops::gdn_input_proj_column_parallel(tensors(2, 1, x_buffer.data(), kHidden),
                                            weights.prepared,
                                            tensors(2, 1, qkv_buffer.data(), kShardChannels),
                                            tensors(2, 1, z_buffer.data(), kShardValueRows), ec);
    });
    failures += expect_invalid("single-device context", [&] {
        const ExecutionContext single({0});
        ops::gdn_input_proj_column_parallel(x, weights.prepared, qkv, z, single);
    });
    failures += expect_invalid("parent-sized weight", [&] {
        ops::gdn_input_proj_column_parallel(x, {weights.parent.weight, weights.parent.weight}, qkv,
                                            z, ec);
    });
    failures += expect_invalid("A8 without workspace", [&] {
        const auto x8   = tensors(8, 8, x_buffer.data(), kHidden);
        const auto qkv8 = tensors(8, 8, qkv_buffer.data(), kShardChannels);
        const auto z8   = tensors(8, 8, z_buffer.data(), kShardValueRows);
        ops::gdn_input_proj_column_parallel(x8, weights.prepared, qkv8, z8,
                                            ops::LinearPolicy::AllowA8, {nullptr, nullptr}, ec);
    });
    failures += expect_invalid("NVFP4 weight", [&] {
        Weight nvfp4 = weights.prepared[0];
        nvfp4.qtype  = QType::NVFP4;
        ops::gdn_input_proj_column_parallel(x, {nvfp4, nvfp4}, qkv, z, ec);
    });
    std::cout << (failures ? "FAIL" : "OK") << " split rejections\n";
    return failures;
}

} // namespace

int main() {
    int failures = verify_registry();
    if (failures != 0) {
        std::cout << "FAIL gdn projections split (registry)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: GDN projection split parity requires two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    const ExecutionContext ec({0, 1});
    InputWeights weights;
    failures += make_input_weights(ec, 45U, weights);
    if (failures == 0) {
        failures += verify_split_rejections(ec, weights);
        failures += run_input_case(ec, weights, 45U);
        // The conv forms take A8 from W=10 at B=1 and from B*W=9 when batched, not from T=8 as the
        // bare projection does; W=8..10 and B*W=8..9 straddle both frontiers.
        const std::vector<ConvCase> snapshot_cases{{1, 1, false},  {1, 3, false}, {1, 4, false},
                                                   {1, 8, false},  {1, 9, false}, {1, 10, false},
                                                   {1, 24, false}, {2, 4, true},  {3, 3, false},
                                                   {3, 4, true},   {8, 2, false}, {2, 16, true}};
        const std::vector<ConvCase> record_cases{
            {1, 4, false}, {1, 9, false}, {2, 4, false}, {3, 5, true}, {2, 16, false}};
        std::uint32_t seed = 61U;
        for (const ops::LinearPolicy policy :
             {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
            for (const ConvCase& test_case : snapshot_cases) {
                failures += run_conv_case(ec, weights, test_case, policy, false, seed++);
            }
            for (const ConvCase& test_case : record_cases) {
                failures += run_conv_case(ec, weights, test_case, policy, true, seed++);
            }
        }
    }
    failures += run_gating_case(ec, 51U, false);
    failures += run_gating_case(ec, 53U, true);

    std::cout << (failures ? "FAIL" : "OK") << " gdn projections split\n";
    return failures ? 1 : 0;
}
