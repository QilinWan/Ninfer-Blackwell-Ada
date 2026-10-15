#include "core/paged_kv_storage.h"
#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // Storage layout table: the rk4v4-e8 row and the regression guards for every existing row.
    // B/tok for one D256 layer = KV heads x physical_bytes_per_token_head; the per-token figures
    // quoted for the product shapes come from these row bytes times the layer count.
    const auto e8 = ninfer::paged_kv_storage_layout(ninfer::KvCacheStorage::RK4V4E8, 256);
    failures +=
        check(e8.key == e8.value && e8.key.data_dtype == ninfer::DType::U8 &&
                  e8.key.data_leading_extent == 128 && e8.key.scale_dtype == ninfer::DType::FP16 &&
                  e8.key.scale_leading_extent == 1 && e8.planes_per_layer() == 4 &&
                  e8.physical_bytes_per_token_head() == 260,
              "rk4v4-e8 row is not two 128-byte i4 code planes plus one FP16 scale per row");

    const std::size_t expected_rows[] = {1024, 528, 516, 288, 402};
    const ninfer::KvCacheStorage rows[] = {
        ninfer::KvCacheStorage::BFloat16,     ninfer::KvCacheStorage::Int8Group64,
        ninfer::KvCacheStorage::Fp8E4M3Row256, ninfer::KvCacheStorage::Nvfp4Group16,
        ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
    };
    for (std::size_t row = 0; row < std::size(rows); ++row) {
        const auto layout = ninfer::paged_kv_storage_layout(rows[row], 256);
        failures += check(layout.physical_bytes_per_token_head() == expected_rows[row],
                          "existing KV storage row changed its D256 physical bytes per token");
    }

    bool e8_geometry_rejected = false;
    try {
        (void)ninfer::paged_kv_storage_layout(ninfer::KvCacheStorage::RK4V4E8, 128);
    } catch (const std::invalid_argument&) { e8_geometry_rejected = true; }
    failures += check(e8_geometry_rejected,
                      "rk4v4-e8 accepted a head dimension outside the D256 KV contract");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
