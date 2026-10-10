#include "serve/prefill_miss_dump.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using namespace ninfer::serve;
using Tokens = std::vector<ninfer::TokenId>;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

Tokens iota_tokens(std::size_t count, ninfer::TokenId first) {
    Tokens tokens(count);
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
}

Tokens concat(Tokens a, const Tokens& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

std::string fake_detokenize(std::span<const ninfer::TokenId> tokens) {
    std::string text;
    for (const ninfer::TokenId token : tokens) { text += std::to_string(token) + ' '; }
    return text;
}

} // namespace

int main() {
    int failures = 0;
    const Tokens a = iota_tokens(10, 0);
    failures += check(common_prefix_length(a, a) == 10, "common prefix of identical sequences");
    failures += check(common_prefix_length(a, Tokens{0, 1, 9}) == 2, "common prefix stops early");
    failures += check(common_prefix_length(a, Tokens{}) == 0, "common prefix with empty");

    const std::filesystem::path root = std::filesystem::temp_directory_path() /
                                       ("ninfer-prefill-miss-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    PrefillMissDump dump(PrefillMissDumpOptions{.directory = root}, fake_detokenize);

    // 1. First request: nothing to compare against.
    const Tokens first        = iota_tokens(20000, 0);
    const Tokens first_output = iota_tokens(500, 900000);
    failures += check(!dump.observe(first, first_output, 0, "finished").has_value(),
                      "first request dumped without a previous context");

    // 2. Same 15000-token head, then 5000 different tokens, nothing reused: the server
    //    recomputed 15000 tokens identical to the previous context.
    const Tokens second = concat(iota_tokens(15000, 0), iota_tokens(5000, 500000));
    const auto miss     = dump.observe(second, {}, 0, "cancelled");
    failures += check(miss.has_value(), "shared-prefix recomputation not reported");
    if (miss.has_value()) {
        failures += check(miss->common_prefix == 15000, "wrong common prefix");
        failures += check(miss->reused_tokens == 0 && miss->prompt_tokens == 20000,
                          "wrong reused/prompt tokens");
        failures += check(miss->old_context_tokens == 20500, "old context lacks the output");
        const std::filesystem::path directory = miss->directory;
        failures += check(std::filesystem::exists(directory / "new_prompt.txt") &&
                              std::filesystem::exists(directory / "old_context.txt") &&
                              std::filesystem::exists(directory / "new_prompt.tokens.json") &&
                              std::filesystem::exists(directory / "old_context.tokens.json"),
                          "dump files missing");
        std::ifstream in(directory / "summary.json");
        const nlohmann::json summary = nlohmann::json::parse(in);
        failures += check(summary.at("common_prefix_tokens") == 15000, "summary common prefix");
        failures += check(summary.at("recomputed_shared_tokens") == 15000,
                          "summary recomputed shared tokens");
        failures += check(summary.at("divergence").at("inside_old_prompt") == true,
                          "summary divergence position");
        failures += check(summary.at("divergence").at("new_after").get<std::string>().rfind(
                              "500000 ", 0) == 0,
                          "summary new text after the divergence");
        failures += check(summary.at("reading").dump().find("server:") !=
                              std::string::npos,
                          "summary reading for a server-side miss");
    }

    // 3. Appending 12000 new tokens to the previous prompt with all of it reused is not a miss.
    const Tokens third        = concat(second, iota_tokens(12000, 700000));
    const Tokens third_output = iota_tokens(500, 950000);
    failures += check(!dump.observe(third, third_output, static_cast<std::uint32_t>(second.size()),
                                    "finished")
                           .has_value(),
                      "appended-only prompt reported as a miss");

    // 4. Under half of the prompt shared with any context: not compared.
    const Tokens fourth = concat(iota_tokens(2000, 0), iota_tokens(30000, 800000));
    failures += check(!dump.observe(fourth, {}, 2000, "finished").has_value(),
                      "a prompt sharing under half of its tokens was dumped");

    // 5. Agent turn: the previous output (thinking) is re-rendered differently and a 15300-token
    //    tool result follows, everything shared reused: a long append, not a miss.
    const Tokens fifth = concat(third, iota_tokens(15300, 300000));
    failures += check(!dump.observe(fifth, {}, static_cast<std::uint32_t>(third.size()),
                                    "finished")
                           .has_value(),
                      "a long tool result after a re-rendered output was dumped");

    // 6. The client rewrites history from token 16000 on (31300 tokens of the closest context
    //    after the divergence), the server reuses everything shared: client-side change.
    const Tokens sixth = concat(Tokens(third.begin(), third.begin() + 16000),
                                iota_tokens(14000, 400000));
    const auto changed = dump.observe(sixth, {}, 16000, "finished");
    failures += check(changed.has_value(), "client-side rewrite not reported");
    if (changed.has_value()) {
        failures += check(changed->common_prefix == 16000 && changed->old_context_tokens == 47300,
                          "client-side rewrite compared against the wrong context");
        std::ifstream in(std::filesystem::path(changed->directory) / "summary.json");
        const nlohmann::json summary = nlohmann::json::parse(in);
        failures += check(summary.at("old_tokens_after_divergence") == 31300,
                          "summary old tokens after the divergence");
        failures += check(summary.at("reading").dump().find("client:") !=
                              std::string::npos,
                          "summary reading for a client-side change");
    }

    std::filesystem::remove_all(root);
    if (failures == 0) { std::cout << "OK prefill miss dump\n"; }
    return failures == 0 ? 0 : 1;
}
