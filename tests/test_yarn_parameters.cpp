#include "ninfer/ops/rope.h"
#include "runtime/contract/yarn.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
// The registered rotary domains other than D256/R64: the D128/R128 full-head text route and the
// D72/R72 vision route, plus a partial-rotary shape no registered route covers.
constexpr std::pair<std::uint32_t, std::uint32_t> kForeignRotaryDomains[]{
    {128U, 128U}, {72U, 72U}, {256U, 128U}};
} // namespace

int main() {
    using namespace ninfer;
    int failures     = 0;
    const auto check = [&](bool ok, const char* label) {
        if (!ok) {
            ++failures;
            std::cerr << label << '\n';
        }
    };
    std::ifstream stream(std::string(NINFER_SOURCE_DIR) + "/tests/data/qwen3_8_yarn_hf.json");
    const auto fixture = nlohmann::json::parse(stream);
    for (const auto& row : fixture.at("cases")) {
        const float factor      = row.at("factor").get<float>();
        const auto coefficients = ops::make_text_yarn_scaling(factor, 262144);
        check(coefficients.enabled == (factor > 1.0F), "native/scaled mode mismatch");
        check(std::abs(coefficients.attention_factor - row.at("attention_factor").get<double>()) <
                  1e-7,
              "YaRN amplitude differs from the Hugging Face reference");
        if (factor > 1.0F) {
            for (int i = 0; i < 32; ++i) {
                const double want = row.at("inverse_frequency").at(i).get<double>();
                check(std::abs(coefficients.inverse_frequency[i] / want - 1.0) < 3e-7,
                      "YaRN frequency differs from the Hugging Face reference");
            }
        }
    }
    for (float factor : {0.0F, 0.9F, 4.1F, std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN()}) {
        bool rejected = false;
        try {
            (void)ops::make_text_yarn_scaling(factor, 262144);
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "invalid YaRN factor accepted");
    }
    runtime::validate_yarn({1.5F, 262144}, 393216);
    bool rejected = false;
    try {
        runtime::validate_yarn({1.5F, 262144}, 393217);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "context above the scaled window accepted");
    rejected = false;
    try {
        runtime::validate_yarn({1.5F, 1048576}, 393216);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "compiled address ceiling mistaken for the native reference window");
    // The coefficient table only expresses the D256/R64 text rotary domain (ops/rope.h). A factor
    // requested for a domain that rotates natively by contract must be answered with that reason at
    // load time, not built and then dropped by the kernel owning the shape.
    runtime::validate_yarn_rotary_domain(256, 64, 1.0F);
    runtime::validate_yarn_rotary_domain(256, 64, 4.0F);
    // Native rope needs no domain it can scale, so a factor of 1 stays valid on every shape.
    runtime::validate_yarn_rotary_domain(128, 128, 1.0F);
    runtime::validate_yarn_rotary_domain(72, 72, 1.0F);
    for (const auto& [head_dim, rotary_dim] : kForeignRotaryDomains) {
        rejected = false;
        try {
            runtime::validate_yarn_rotary_domain(head_dim, rotary_dim, 1.5F);
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "YaRN factor accepted outside the D256/R64 rotary domain");
    }
    check(runtime::yarn_cache_binding({1.0F, 262144}).empty(), "native snapshot binding changed");
    check(runtime::yarn_cache_binding({1.5F, 262144}) !=
              runtime::yarn_cache_binding({2.0F, 262144}),
          "different positional transforms can alias a retained KV snapshot");
    std::cout << (failures ? "FAIL" : "OK") << " YaRN CPU reference and cache contract\n";
    return failures ? 1 : 0;
}
