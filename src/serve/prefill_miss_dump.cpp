#include "serve/prefill_miss_dump.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <utility>

namespace ninfer::serve {
namespace {

// Tokens of context shown on each side of the divergence point in summary.json.
constexpr std::size_t kSnippetTokens = 96;
// A process writes at most this many dumps, so a systematic miss cannot fill the disk.
constexpr std::uint64_t kMaximumDumps = 64;

std::string utc_stamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
    return out.str();
}

std::span<const TokenId> clamp_span(std::span<const TokenId> tokens, std::size_t begin,
                                    std::size_t end) {
    begin = std::min(begin, tokens.size());
    end   = std::clamp(end, begin, tokens.size());
    return tokens.subspan(begin, end - begin);
}

void write_text(const std::filesystem::path& path, std::string_view text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) { throw std::runtime_error("cannot write " + path.string()); }
}

} // namespace

std::size_t common_prefix_length(std::span<const TokenId> a, std::span<const TokenId> b) noexcept {
    const std::size_t limit = std::min(a.size(), b.size());
    std::size_t length      = 0;
    while (length < limit && a[length] == b[length]) { ++length; }
    return length;
}

PrefillMissDump::PrefillMissDump(PrefillMissDumpOptions options, Detokenizer detokenize)
    : options_(std::move(options)), detokenize_(std::move(detokenize)) {
    std::filesystem::create_directories(options_.directory);
}

std::optional<PrefillMissReport> PrefillMissDump::observe(std::span<const TokenId> prompt,
                                                          std::span<const TokenId> output,
                                                          std::uint32_t reused_tokens,
                                                          std::string_view label) {
    std::lock_guard lock(mutex_);
    std::optional<PrefillMissReport> report;
    const std::size_t reused  = std::min<std::size_t>(reused_tokens, prompt.size());
    const std::size_t prefill = prompt.size() - reused;
    if (prefill >= options_.min_tokens && !contexts_.empty()) {
        const Context* closest = nullptr;
        std::size_t best       = 0;
        // Most recent first, so ties go to the latest context.
        for (auto it = contexts_.rbegin(); it != contexts_.rend(); ++it) {
            const std::size_t length = common_prefix_length(prompt, it->tokens);
            if (closest == nullptr || length > best) {
                closest = &*it;
                best    = length;
            }
        }
        const bool matches = static_cast<double>(best) >=
                             options_.min_match * static_cast<double>(prompt.size());
        const std::size_t server_recomputed = best > reused ? best - reused : 0;
        const std::size_t client_rewritten =
            closest->tokens.size() > best ? closest->tokens.size() - best : 0;
        const bool suspicious = server_recomputed >= options_.min_tokens ||
                                client_rewritten >= options_.min_tokens;
        if (matches && suspicious && sequence_ < kMaximumDumps) {
            report = PrefillMissReport{
                .prompt_tokens      = static_cast<std::uint32_t>(prompt.size()),
                .reused_tokens      = static_cast<std::uint32_t>(reused),
                .common_prefix      = static_cast<std::uint32_t>(best),
                .old_context_tokens = static_cast<std::uint32_t>(closest->tokens.size()),
            };
            try {
                report->directory = write_dump(prompt, static_cast<std::uint32_t>(reused), label,
                                               *closest, best);
            } catch (const std::exception&) { report->directory.clear(); }
        }
    }
    Context context;
    context.tokens.reserve(prompt.size() + output.size());
    context.tokens.assign(prompt.begin(), prompt.end());
    context.tokens.insert(context.tokens.end(), output.begin(), output.end());
    context.prompt_tokens = static_cast<std::uint32_t>(prompt.size());
    context.label         = std::string(label);
    contexts_.push_back(std::move(context));
    while (contexts_.size() > options_.history) { contexts_.pop_front(); }
    return report;
}

std::string PrefillMissDump::write_dump(std::span<const TokenId> prompt,
                                        std::uint32_t reused_tokens, std::string_view label,
                                        const Context& old, std::size_t common_prefix) {
    const std::uint64_t sequence = ++sequence_;
    std::ostringstream name;
    name << utc_stamp() << '-' << std::setw(3) << std::setfill('0') << sequence;
    const std::filesystem::path directory = options_.directory / name.str();
    std::filesystem::create_directories(directory);

    const std::span<const TokenId> old_tokens(old.tokens);
    write_text(directory / "new_prompt.txt", detokenize_(prompt));
    write_text(directory / "old_context.txt", detokenize_(old_tokens));
    write_text(directory / "new_prompt.tokens.json",
               nlohmann::json(std::vector<TokenId>(prompt.begin(), prompt.end())).dump());
    write_text(directory / "old_context.tokens.json", nlohmann::json(old.tokens).dump());

    const std::size_t shared_recomputed =
        common_prefix > reused_tokens ? common_prefix - reused_tokens : 0;
    const std::size_t snippet_begin =
        common_prefix > kSnippetTokens ? common_prefix - kSnippetTokens : 0;
    nlohmann::json summary{
        {"new_request", std::string(label)},
        {"new_prompt_tokens", prompt.size()},
        {"reused_tokens", reused_tokens},
        {"recomputed_tokens", prompt.size() - reused_tokens},
        {"old_request", old.label},
        {"old_context_tokens", old.tokens.size()},
        {"old_prompt_tokens", old.prompt_tokens},
        {"common_prefix_tokens", common_prefix},
        {"recomputed_shared_tokens", shared_recomputed},
        {"old_tokens_after_divergence", old.tokens.size() - common_prefix},
        {"divergence",
         {{"token_index", common_prefix},
          {"inside_old_prompt", common_prefix < old.prompt_tokens},
          {"shared_before", detokenize_(clamp_span(prompt, snippet_begin, common_prefix))},
          {"new_after",
           detokenize_(clamp_span(prompt, common_prefix, common_prefix + kSnippetTokens))},
          {"old_after",
           detokenize_(clamp_span(old_tokens, common_prefix, common_prefix + kSnippetTokens))}}},
    };
    // Both readings can hold at once: checkpoints are discrete, so a client-side change can also
    // leave identical tokens to recompute before it.
    nlohmann::json reading = nlohmann::json::array();
    const std::size_t old_after = old.tokens.size() - common_prefix;
    if (old_after >= options_.min_tokens) {
        reading.push_back("client: the request changed or dropped " + std::to_string(old_after) +
                          " tokens of the earlier context after the divergence point");
    }
    if (shared_recomputed >= options_.min_tokens) {
        reading.push_back("server: " + std::to_string(shared_recomputed) +
                          " recomputed tokens are identical to the earlier context; the "
                          "closest usable checkpoint was at reused_tokens");
    }
    summary["reading"] = std::move(reading);
    write_text(directory / "summary.json", summary.dump(2) + "\n");
    return directory.string();
}

} // namespace ninfer::serve
