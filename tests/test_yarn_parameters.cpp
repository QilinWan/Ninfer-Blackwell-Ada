#include "ninfer/ops/rope.h"
#include "runtime/contract/yarn.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

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
    check(runtime::yarn_cache_binding({1.0F, 262144}).empty(), "native snapshot binding changed");
    check(runtime::yarn_cache_binding({1.5F, 262144}) !=
              runtime::yarn_cache_binding({2.0F, 262144}),
          "different positional transforms can alias a retained KV snapshot");
    std::cout << (failures ? "FAIL" : "OK") << " YaRN CPU reference and cache contract\n";
    return failures ? 1 : 0;
}
