#pragma once

// --prefill-miss-dump: a diagnostic for prefix-reuse misses. Serve remembers the token contexts of
// its recent requests (prompt plus committed output, cancelled requests included). A request that
// recomputes at least min_tokens prompt tokens, and whose longest common prefix with the closest of
// those contexts covers at least min_match of its prompt, is dumped when either
//   - the server recomputed at least min_tokens tokens identical to that context (it resumed from
//     an earlier checkpoint than the shared prefix allowed), or
//   - at least min_tokens tokens of that context lie after the divergence (the client changed or
//     dropped earlier text, e.g. re-rendered history).
// A request that only appends new text, however long, is not dumped.
//
// The dumps contain conversation text verbatim. Off unless the option is given.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

inline constexpr std::uint32_t kDefaultPrefillMissMinTokens = 10000;
inline constexpr double kDefaultPrefillMissMinMatch         = 0.5;
inline constexpr std::size_t kPrefillMissHistory            = 16;

struct PrefillMissDumpOptions {
    std::filesystem::path directory;
    std::uint32_t min_tokens = kDefaultPrefillMissMinTokens;
    double min_match         = kDefaultPrefillMissMinMatch;
    std::size_t history      = kPrefillMissHistory;
};

// One detected miss, reported in the operational log next to the request's done line.
struct PrefillMissReport {
    std::string directory;
    std::uint32_t prompt_tokens     = 0;
    std::uint32_t reused_tokens     = 0;
    std::uint32_t common_prefix      = 0; // longest common prefix with the closest recent context
    std::uint32_t old_context_tokens = 0;
};

// Length of the longest common prefix of two token sequences.
[[nodiscard]] std::size_t common_prefix_length(std::span<const TokenId> a,
                                               std::span<const TokenId> b) noexcept;

class PrefillMissDump {
public:
    using Detokenizer = std::function<std::string(std::span<const TokenId>)>;

    PrefillMissDump(PrefillMissDumpOptions options, Detokenizer detokenize);

    // Called once per finished request (any outcome). Checks the request against the remembered
    // contexts, writes a dump when it qualifies, then remembers prompt + output as a context.
    // `label` names the request in the dump (e.g. its finish reason). Never throws: a failed dump
    // is reported through the returned report's directory being empty.
    std::optional<PrefillMissReport> observe(std::span<const TokenId> prompt,
                                             std::span<const TokenId> output,
                                             std::uint32_t reused_tokens, std::string_view label);

private:
    struct Context {
        std::vector<TokenId> tokens;
        std::uint32_t prompt_tokens = 0;
        std::string label;
    };

    std::string write_dump(std::span<const TokenId> prompt, std::uint32_t reused_tokens,
                           std::string_view label, const Context& old,
                           std::size_t common_prefix);

    PrefillMissDumpOptions options_;
    Detokenizer detokenize_;
    std::mutex mutex_;
    std::deque<Context> contexts_;
    std::uint64_t sequence_ = 0;
};

} // namespace ninfer::serve
