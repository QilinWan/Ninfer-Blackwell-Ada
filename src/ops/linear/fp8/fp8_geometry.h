#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
template <std::int32_t OutputRows, std::int32_t InputRows>
struct Fp8Geometry {
    static_assert(OutputRows > 0 && (OutputRows % 16) == 0);
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <std::int32_t InputRows>
struct Fp8ActivationGeometry {
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kInputRows = InputRows;
};

using Fp8N14336K5120             = Fp8Geometry<14336, 5120>;
using Fp8N16384K5120             = Fp8Geometry<16384, 5120>;
using Fp8N34816K5120             = Fp8Geometry<34816, 5120>;
using Fp8N248320K5120            = Fp8Geometry<248320, 5120>;
using Fp8N5120K6144              = Fp8Geometry<5120, 6144>;
using Fp8N5120K17408             = Fp8Geometry<5120, 17408>;
// Two-device shards: output-row halves of the input projections and the vocabulary head, and
// input-column halves of the residual projections. Each inherits the routes of the parent.
using Fp8N7168K5120              = Fp8Geometry<7168, 5120>;
using Fp8N8192K5120              = Fp8Geometry<8192, 5120>;
using Fp8N17408K5120             = Fp8Geometry<17408, 5120>;
using Fp8N124160K5120            = Fp8Geometry<124160, 5120>;
using Fp8N5120K3072              = Fp8Geometry<5120, 3072>;
using Fp8N5120K8704              = Fp8Geometry<5120, 8704>;

// First token count at which fp8 linear_add over the attention output ([5120,6144] and its
// two-device half [5120,3072]) and over the MLP down projection ([5120,17408] and [5120,8704])
// takes the A8 route. linear() over each half uses the same crossover, so rank 0's linear_add
// and rank 1's linear() in one row-parallel pair always take the same route.
// This tree's measured frontiers (the fork's 22 / 25 are its own 5070 Ti tuning): 17 for the
// [5120,6144] attention-output family and 20 for the MLP down family; the two-device halves
// inherit the frontier of the family they halve, so rank 0's linear_add and rank 1's linear() in
// one row-parallel pair always take the same route.
inline constexpr std::int32_t kFp8OutputFamilyFirstA8Tokens = 17;
inline constexpr std::int32_t kFp8DownFamilyFirstA8Tokens   = 20;
using Fp8Activation3072Geometry  = Fp8ActivationGeometry<3072>;
using Fp8Activation5120Geometry  = Fp8ActivationGeometry<5120>;
using Fp8Activation6144Geometry  = Fp8ActivationGeometry<6144>;
using Fp8Activation8704Geometry  = Fp8ActivationGeometry<8704>;
using Fp8Activation17408Geometry = Fp8ActivationGeometry<17408>;

enum class Fp8GeometryId : std::uint8_t {
    N14336K5120,
    N16384K5120,
    N34816K5120,
    N248320K5120,
    N5120K6144,
    N5120K17408,
    // Two-device shards: output-row halves of the input projections and the vocabulary head,
    // input-column halves of the residual projections.
    N7168K5120,
    N8192K5120,
    N17408K5120,
    N124160K5120,
    N5120K3072,
    N5120K8704,
};

inline Fp8GeometryId resolve_fp8_geometry(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Fp8N14336K5120::kOutputRows && input_rows == Fp8N14336K5120::kInputRows) {
        return Fp8GeometryId::N14336K5120;
    }
    if (output_rows == Fp8N16384K5120::kOutputRows && input_rows == Fp8N16384K5120::kInputRows) {
        return Fp8GeometryId::N16384K5120;
    }
    if (output_rows == Fp8N34816K5120::kOutputRows && input_rows == Fp8N34816K5120::kInputRows) {
        return Fp8GeometryId::N34816K5120;
    }
    if (output_rows == Fp8N248320K5120::kOutputRows && input_rows == Fp8N248320K5120::kInputRows) {
        return Fp8GeometryId::N248320K5120;
    }
    if (output_rows == Fp8N5120K6144::kOutputRows && input_rows == Fp8N5120K6144::kInputRows) {
        return Fp8GeometryId::N5120K6144;
    }
    if (output_rows == Fp8N5120K17408::kOutputRows && input_rows == Fp8N5120K17408::kInputRows) {
        return Fp8GeometryId::N5120K17408;
    }
    if (output_rows == Fp8N7168K5120::kOutputRows && input_rows == Fp8N7168K5120::kInputRows) {
        return Fp8GeometryId::N7168K5120;
    }
    if (output_rows == Fp8N8192K5120::kOutputRows && input_rows == Fp8N8192K5120::kInputRows) {
        return Fp8GeometryId::N8192K5120;
    }
    if (output_rows == Fp8N17408K5120::kOutputRows && input_rows == Fp8N17408K5120::kInputRows) {
        return Fp8GeometryId::N17408K5120;
    }
    if (output_rows == Fp8N124160K5120::kOutputRows && input_rows == Fp8N124160K5120::kInputRows) {
        return Fp8GeometryId::N124160K5120;
    }
    if (output_rows == Fp8N5120K3072::kOutputRows && input_rows == Fp8N5120K3072::kInputRows) {
        return Fp8GeometryId::N5120K3072;
    }
    if (output_rows == Fp8N5120K8704::kOutputRows && input_rows == Fp8N5120K8704::kInputRows) {
        return Fp8GeometryId::N5120K8704;
    }
    throw std::invalid_argument("unsupported FP8 problem");
}

} // namespace ninfer::ops::detail
