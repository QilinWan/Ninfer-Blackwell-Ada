// Two-device tensor-parallel (tp 2) Engine with MTP speculative decoding (K=3) on
// NINFER_TEST_ARTIFACT, through the public generation route ninfer-serve uses. Qwen3.8 27B does not
// fit one 16 GB device, so the reference is the same model at tp 2 without speculation. CUDA
// Graphs, the context cache and two request lanes are on, as in the served default. With the
// argument `optimized` the MTP Engine selects the optimized proposal head (rank 0 only) instead of
// the vocabulary-split full head.
//
//   * parity: four greedy prompts (arithmetic, code, a ~200-token text) run first on a tp 2 Engine
//     without speculation and then on the MTP Engine. Greedy verification is lossless with respect
//     to the target's own argmax, so the answers must agree. They are not required to be
//     bit-identical end to end: verification evaluates the target over K+1 columns and ordinary
//     decode over one, which selects different GEMM and attention routes, and greedy decoding
//     amplifies a last-bit logit difference at a near tie (the single-device MTP path has the same
//     property). A near tie can fall anywhere, even on the first content words (measured with the
//     optimized head: "leggendaria" against "mitologica" at token 9 of the free text, identical in
//     meaning afterwards), and the public API exposes no logits to bound the gap. The gate is
//     therefore the checkable answers correct, all but one prompt sharing at least
//     kMinimumCommonPrefix leading tokens, and at least half identical: a broken verification or
//     GDN fold corrupts every prompt within its first rounds, not one.
//   * acceptance: over the parity prompts at least kMinimumAcceptance of the drafted tokens are
//     accepted. A draft side broken by the split (a lost half of the fc input, the proposal logits
//     or the MTP KV) collapses to chance acceptance while the output stays right.
//   * single and concurrent: "What is 17*23?" answers 391 alone and while a long answer holds
//     the other lane, in shared two-row MTP rounds; a second pair runs together as well.
//   * per-step accepted lengths (GenerationObservationOptions::accepted_length_steps): the
//     recorded sequence has one entry per decode step and reproduces rounds, fallback steps,
//     accepted tokens and accepted_per_position exactly; recording leaves the greedy tokens and
//     counters of an unrecorded run unchanged; a cap keeps exactly the first steps; in a shared
//     two-row round only the observed request records.
//   * prepared prompt token ids: PreparedPrompt::token_ids() has summary().prompt_tokens ids of a
//     chat prompt, and a raw-token prompt exposes exactly the ids it was prepared from.
//   * repeated requests: more than the private continuation catalog's worth of requests in series,
//     each binding and releasing both ranks' MTP KV rows (a leaked rank 1 row fails the bind of a
//     later request).
//   * prefix reuse, shaped like ninfer-serve's Chat Completions traffic (no session key, the
//     default implicit shared-prefix marker after the last message), on a 16k-context INT8-KV
//     Engine with 4 lanes and with 1 lane: a ~9k-token document with a question, then the same
//     conversation with the answer and a new question. The second turn resumes a retained prefix
//     through the MTP bridge on both ranks: it must reuse at least 80 % of the first prompt, run
//     MTP rounds and answer correctly. Parity: the same second-turn prompt prefilled cold (prefix
//     reuse disabled, so admission starts from the root exactly as on a fresh Engine) must give
//     the identical greedy answer, and so must an exact repeat of it, run before the cold leg
//     (whose whole-prompt prefill would evict the retained conversation from the 16k KV pool). The
//     repeat resumes the whole prompt when the template retains a response-replay checkpoint at
//     the generation prompt (zero suffix: the first token is sampled from the retained hidden
//     through the vocabulary-split head), else an earlier checkpoint of the second turn. The
//     answer is a three-digit number, far from a near tie. Greedy
//     verification is lossless, so a wrong rank 1 hidden in the bridge would lower acceptance
//     rather than change the answer; the acceptance of the resumed and cold runs is printed for
//     comparison.
//
// Returns 77 without the artifact or below two CUDA devices.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext        = 8192;
constexpr std::uint32_t kLongMaxContext    = 16384;
constexpr std::uint32_t kMinimumLongPrompt = 4096;
constexpr std::uint32_t kDraftTokens       = 3;
constexpr std::size_t kMinimumCommonPrefix = 24;
constexpr double kMinimumAcceptance        = 0.35;
constexpr std::uint32_t kRepeatedRequests  = 14;

ninfer::EngineOptions engine_options(const char* artifact, bool mtp, bool optimized) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.tp                   = 2;
    options.device               = 0;
    options.devices              = {0, 1};
    options.max_context          = kMaxContext;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(2 * kMaxContext);
    options.prefill_chunk        = 1024;
    options.max_concurrency      = 2;
    options.max_pending_requests = 4;
    // Rank 1's KV and StateImages have no Host tier; the Device StateImage and private catalog
    // floors come from the Engine's tp 2 defaults.
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = kDraftTokens;
        options.speculative.proposal_head =
            optimized ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
    }
    return options;
}

// The serving shape of the prefix-reuse legs: 16k context and KV, INT8 KV, `lanes` lanes and the
// Engine's tp 2 context-cache defaults.
ninfer::EngineOptions long_engine_options(const char* artifact, bool optimized,
                                          std::uint32_t lanes) {
    ninfer::EngineOptions options = engine_options(artifact, true, optimized);
    options.max_context           = kLongMaxContext;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(kLongMaxContext);
    options.max_concurrency       = lanes;
    options.max_pending_requests  = 4;
    options.kv_cache              = ninfer::KvCacheStorage::Int8Group64;
    return options;
}

ninfer::PromptInput user_prompt(std::string text) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::RequestOptions greedy(std::uint32_t tokens, bool allow_prefix_reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = allow_prefix_reuse;
    return options;
}

bool contains(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

int check_answer(const ninfer::GenerationResult& result, std::string_view expected,
                 const char* label) {
    if (contains(result.content, expected)) { return 0; }
    std::cerr << label << ": expected \"" << expected << "\" in the answer, got \""
              << result.content << "\" (" << result.generated_token_ids.size() << " tokens)\n";
    return 1;
}

struct Probe {
    const char* label;
    const char* prompt;
    std::uint32_t tokens;
    const char* expected; // Checked in both answers; empty for free text.
};

const std::vector<Probe>& probes() {
    static const std::vector<Probe> list{
        {"product", "What is 17*23? Answer with the number only.", 32, "391"},
        {"square", "What is 12*12? Answer with the number only.", 32, "144"},
        {"code",
         "Write an iterative Python function that returns the factorial of n. Reply with the code "
         "only.",
         128, "def "},
        {"text",
         "Write a paragraph of about 150 words on the founding of Rome, with no headings or "
         "lists.",
         200, ""},
    };
    return list;
}

struct ProbeRun {
    std::vector<ninfer::TokenId> tokens;
    std::string content;
};

std::vector<ProbeRun> run_probes(ninfer::Engine& engine) {
    std::vector<ProbeRun> runs;
    for (const Probe& probe : probes()) {
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare(user_prompt(probe.prompt)), greedy(probe.tokens));
        runs.push_back({result.generated_token_ids, result.content});
    }
    return runs;
}

std::size_t common_prefix(const std::vector<ninfer::TokenId>& a,
                          const std::vector<ninfer::TokenId>& b) {
    const std::size_t limit = std::min(a.size(), b.size());
    std::size_t index       = 0;
    while (index < limit && a[index] == b[index]) { ++index; }
    return index;
}

int exercise_parity(ninfer::Engine& engine, const std::vector<ProbeRun>& reference) {
    int failures           = 0;
    std::size_t identical  = 0;
    std::size_t prefixed   = 0;
    std::uint64_t drafted  = 0;
    std::uint64_t accepted = 0;
    const auto& probe_list = probes();
    for (std::size_t index = 0; index < probe_list.size(); ++index) {
        const Probe& probe = probe_list[index];
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare(user_prompt(probe.prompt)), greedy(probe.tokens));
        drafted += result.speculative.drafted_tokens;
        accepted += result.speculative.accepted_tokens;
        const ProbeRun& base     = reference[index];
        const std::size_t shared = common_prefix(result.generated_token_ids, base.tokens);
        const bool same          = result.generated_token_ids == base.tokens;
        const std::size_t minimum =
            std::min({kMinimumCommonPrefix, result.generated_token_ids.size(), base.tokens.size()});
        identical += same ? 1U : 0U;
        std::cout << "parity " << probe.label << ": " << (same ? "identical" : "diverged") << ", "
                  << shared << "/" << base.tokens.size() << " common tokens, acceptance "
                  << result.speculative.accepted_tokens << "/" << result.speculative.drafted_tokens
                  << '\n';
        if (result.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
            result.speculative.rounds == 0) {
            std::cerr << "parity " << probe.label << ": no MTP round ran\n";
            ++failures;
        }
        if (shared >= minimum) {
            ++prefixed;
        } else {
            std::cout << "parity " << probe.label << ": diverged after " << shared
                      << " tokens\n  MTP:      \"" << result.content << "\"\n  ordinary: \""
                      << base.content << "\"\n";
        }
        if (*probe.expected != '\0') {
            failures += check_answer(result, probe.expected, probe.label);
            if (!contains(base.content, probe.expected)) {
                std::cerr << "parity " << probe.label << ": the ordinary reference is wrong\n";
                ++failures;
            }
        }
    }
    if (prefixed + 1 < probe_list.size()) {
        std::cerr << "parity: only " << prefixed << " of " << probe_list.size()
                  << " answers share their first " << kMinimumCommonPrefix
                  << " tokens with the ordinary tp 2 answers\n";
        ++failures;
    }
    if (2 * identical < probe_list.size()) {
        std::cerr << "parity: only " << identical << " of " << probe_list.size()
                  << " answers are identical to the ordinary tp 2 answers\n";
        ++failures;
    }
    const double acceptance =
        drafted == 0 ? 0.0 : static_cast<double>(accepted) / static_cast<double>(drafted);
    std::cout << "MTP acceptance over the parity prompts: " << accepted << "/" << drafted << " ("
              << acceptance << ")\n";
    if (acceptance < kMinimumAcceptance) {
        std::cerr << "MTP acceptance " << acceptance << " is below " << kMinimumAcceptance << '\n';
        ++failures;
    }
    return failures;
}

int exercise_concurrent(ninfer::Engine& engine) {
    int failures                      = 0;
    const ninfer::RuntimeStats before = engine.runtime_stats();
    ninfer::GenerationHandle counting = engine.submit(
        engine.prepare(user_prompt("Write the integers from 1 to 40, separated by a comma and a "
                                   "space, and nothing else.")),
        greedy(160));
    ninfer::GenerationHandle product = engine.submit(
        engine.prepare(user_prompt("What is 17*23? Answer with the number only.")), greedy(32));
    const ninfer::GenerationResult counted    = counting.wait();
    const ninfer::GenerationResult multiplied = product.wait();
    const ninfer::RuntimeStats after          = engine.runtime_stats();
    failures += check_answer(counted, "38, 39, 40", "concurrent long lane");
    failures += check_answer(multiplied, "391", "concurrent second lane");
    const std::uint64_t rounds = after.decode_rounds - before.decode_rounds;
    const std::uint64_t rows   = after.decode_row_rounds - before.decode_row_rounds;
    if (rounds == 0 || rows <= rounds) {
        std::cerr << "concurrent leg ran no two-row MTP round (" << rounds << " rounds, " << rows
                  << " rows)\n";
        ++failures;
    }
    if (counted.speculative.accepted_tokens == 0) {
        std::cerr << "concurrent long lane accepted no draft\n";
        ++failures;
    }

    ninfer::GenerationHandle first = engine.submit(
        engine.prepare(user_prompt("What is 17*23? Answer with the number only.")), greedy(32));
    ninfer::GenerationHandle second = engine.submit(
        engine.prepare(user_prompt("What is 12*12? Answer with the number only.")), greedy(32));
    failures += check_answer(first.wait(), "391", "concurrent pair, first");
    failures += check_answer(second.wait(), "144", "concurrent pair, second");
    return failures;
}

// Rebuilds the counters from a complete per-step sequence and compares them with the Engine's.
int check_accepted_lengths(const ninfer::SpeculativeStats& stats, const char* label) {
    std::uint64_t rounds = 0, fallback = 0, accepted = 0;
    std::vector<std::uint64_t> at_least(stats.draft_window, 0);
    for (const std::uint8_t length : stats.accepted_lengths) {
        if (length == ninfer::kSpeculativeFallbackStep) {
            ++fallback;
            continue;
        }
        if (length > stats.draft_window) {
            std::cerr << label << ": accepted length " << int{length} << " exceeds the window\n";
            return 1;
        }
        ++rounds;
        accepted += length;
        for (std::uint8_t i = 0; i < length; ++i) { ++at_least[i]; }
    }
    if (rounds != stats.rounds || fallback != stats.fallback_steps ||
        accepted != stats.accepted_tokens || at_least != stats.accepted_per_position) {
        std::cerr << label << ": " << stats.accepted_lengths.size()
                  << " recorded steps do not reproduce the counters (rounds " << rounds << "/"
                  << stats.rounds << ", fallback " << fallback << "/" << stats.fallback_steps
                  << ", accepted " << accepted << "/" << stats.accepted_tokens << ")\n";
        return 1;
    }
    return 0;
}

int exercise_accepted_lengths(ninfer::Engine& engine) {
    int failures               = 0;
    const char* const prompt   = probes().back().prompt;
    const std::uint32_t tokens = probes().back().tokens;
    const auto run             = [&](std::uint32_t steps) {
        return engine
            .submit(engine.prepare(user_prompt(prompt)), greedy(tokens),
                    ninfer::OutputConsumerMode::Aggregate,
                    ninfer::GenerationObservationOptions{.accepted_length_steps = steps})
            .wait();
    };
    const ninfer::GenerationResult plain    = run(0);
    const ninfer::GenerationResult recorded = run(4096);
    const ninfer::GenerationResult capped   = run(5);
    const ninfer::SpeculativeStats& stats   = recorded.speculative;
    std::cout << "accepted lengths: " << stats.accepted_lengths.size() << " steps, " << stats.rounds
              << " rounds, " << stats.fallback_steps << " fallback steps\n";
    if (!plain.speculative.accepted_lengths.empty()) {
        std::cerr << "accepted lengths were recorded without the observation\n";
        ++failures;
    }
    if (stats.rounds < 8) {
        std::cerr << "accepted lengths: only " << stats.rounds << " rounds ran\n";
        ++failures;
    }
    failures += check_accepted_lengths(stats, "accepted lengths");
    if (recorded.generated_token_ids != plain.generated_token_ids ||
        stats.rounds != plain.speculative.rounds ||
        stats.accepted_tokens != plain.speculative.accepted_tokens ||
        stats.accepted_per_position != plain.speculative.accepted_per_position) {
        std::cerr << "recording accepted lengths changed the greedy run\n";
        ++failures;
    }
    if (capped.speculative.accepted_lengths.size() != 5 ||
        !std::equal(capped.speculative.accepted_lengths.begin(),
                    capped.speculative.accepted_lengths.end(), stats.accepted_lengths.begin())) {
        std::cerr << "a 5-step cap did not keep exactly the first 5 steps\n";
        ++failures;
    }

    // Shared two-row rounds: only the observed lane records, and its sequence stays its own.
    ninfer::GenerationHandle counting = engine.submit(
        engine.prepare(user_prompt("Write the integers from 1 to 40, separated by a comma and a "
                                   "space, and nothing else.")),
        greedy(160), ninfer::OutputConsumerMode::Aggregate,
        ninfer::GenerationObservationOptions{.accepted_length_steps = 4096});
    ninfer::GenerationHandle product = engine.submit(
        engine.prepare(user_prompt("What is 17*23? Answer with the number only.")), greedy(32));
    const ninfer::GenerationResult counted    = counting.wait();
    const ninfer::GenerationResult multiplied = product.wait();
    failures += check_accepted_lengths(counted.speculative, "concurrent accepted lengths");
    if (!multiplied.speculative.accepted_lengths.empty()) {
        std::cerr << "an unobserved concurrent lane recorded accepted lengths\n";
        ++failures;
    }
    return failures;
}

int exercise_prompt_token_ids(const ninfer::Engine& engine) {
    int failures                      = 0;
    const ninfer::PreparedPrompt chat = engine.prepare(user_prompt(probes().front().prompt));
    const std::span<const ninfer::TokenId> chat_ids = chat.token_ids();
    if (chat_ids.size() != chat.summary().prompt_tokens || chat_ids.empty()) {
        std::cerr << "prepared prompt exposes " << chat_ids.size() << " token ids for "
                  << chat.summary().prompt_tokens << " prompt tokens\n";
        ++failures;
    }
    const std::vector<ninfer::TokenId> ids(chat_ids.begin(), chat_ids.end());
    const ninfer::PreparedPrompt raw = engine.prepare_tokens(ids);
    if (!std::equal(ids.begin(), ids.end(), raw.token_ids().begin(), raw.token_ids().end())) {
        std::cerr << "a raw-token prompt does not expose the ids it was prepared from\n";
        ++failures;
    }
    if (!ninfer::PreparedPrompt{}.token_ids().empty()) {
        std::cerr << "an empty prepared prompt exposes token ids\n";
        ++failures;
    }
    return failures;
}

int exercise_repeated(ninfer::Engine& engine) {
    int failures = 0;
    for (std::uint32_t index = 0; index < kRepeatedRequests; ++index) {
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare(user_prompt("What is " + std::to_string(11 + index) +
                                                       "*3? Answer with the number only.")),
                            greedy(8));
        const std::string expected = std::to_string((11 + index) * 3);
        if (!contains(result.content, expected)) {
            std::cerr << "repeated request " << index << ": expected " << expected << ", got \""
                      << result.content << "\"\n";
            ++failures;
        }
    }
    return failures;
}

// OpenAI Chat Completions without prompt_cache_options: one default implicit shared-prefix
// candidate after the last content part, and no Engine structural candidates
// (serve/openai_common.cpp).
void mark_like_chat_completions(ninfer::PromptInput& input) {
    const ninfer::ChatMessage& last = input.messages.back();
    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count      = static_cast<std::uint32_t>(input.messages.size()),
        .kind                     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence                 = ninfer::SharedCandidateEvidence::DefaultAutomatic,
        .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = static_cast<std::uint32_t>(last.parts.size()),
    });
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
}

std::string long_document() {
    std::string text = "Read the warehouse log, then answer the question.\n\n";
    for (int line = 1; line <= 240; ++line) {
        text += "Line " + std::to_string(line) + ": the Verona warehouse received " +
                std::to_string(line * 7) + " crates of apples and " + std::to_string(line * 3) +
                " crates of pears, shipped on day " + std::to_string(1 + line % 28) + ".\n";
    }
    return text;
}

void print_run(const char* label, const char* leg, const ninfer::GenerationResult& result,
               std::uint64_t computed_prefill_tokens) {
    std::cout << label << ", " << leg << ": reused " << result.reused_prompt_tokens << " of "
              << result.prompt.prompt_tokens << " prompt tokens (path "
              << static_cast<int>(result.prefix_reuse_path) << "), computed prefill "
              << computed_prefill_tokens << ", acceptance " << result.speculative.accepted_tokens
              << "/" << result.speculative.drafted_tokens << ", answer \"" << result.content
              << "\"\n";
}

int exercise_prefix_reuse(ninfer::Engine& engine, const char* label) {
    const std::string document = long_document();
    const auto first_turn      = [&] {
        ninfer::PromptInput input =
            user_prompt(document + "\nQuestion: what is 17*23? Answer with the number only.");
        mark_like_chat_completions(input);
        return input;
    };
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_turn()), greedy(32, true));
    int failures = check_answer(first, "391", label);

    const auto second_turn = [&] {
        ninfer::PromptInput input = first_turn();
        input.context_cache       = {};
        ninfer::ChatMessage assistant;
        assistant.role = ninfer::ChatRole::Assistant;
        assistant.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
        input.messages.push_back(std::move(assistant));
        ninfer::ChatMessage question;
        question.role = ninfer::ChatRole::User;
        question.parts.push_back(
            ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                .text  = "What is 12*12? Answer with the number only.",
                                .media = {}});
        input.messages.push_back(std::move(question));
        mark_like_chat_completions(input);
        return input;
    };
    const auto run = [&](bool allow_prefix_reuse, const char* leg) {
        const ninfer::RuntimeStats before = engine.runtime_stats();
        ninfer::GenerationResult result =
            engine.generate(engine.prepare(second_turn()), greedy(32, allow_prefix_reuse));
        const ninfer::RuntimeStats after = engine.runtime_stats();
        print_run(label, leg, result,
                  after.computed_prefill_tokens - before.computed_prefill_tokens);
        std::cout << "  selections endpoint "
                  << after.private_endpoint_selections - before.private_endpoint_selections
                  << " turn-closure "
                  << after.private_turn_closure_selections - before.private_turn_closure_selections
                  << " anchor "
                  << after.private_long_anchor_selections - before.private_long_anchor_selections
                  << " shared "
                  << after.shared_stable_prefix_selections - before.shared_stable_prefix_selections
                  << "; evicted private "
                  << after.pressure_private_owners_evicted - before.pressure_private_owners_evicted
                  << " shared "
                  << after.pressure_shared_owners_evicted - before.pressure_shared_owners_evicted
                  << " checkpoints dropped "
                  << after.pressure_checkpoints_dropped - before.pressure_checkpoints_dropped
                  << '\n';
        return result;
    };
    // The cold leg runs last: it prefills the whole prompt into pages of its own, and on this
    // 16k KV pool that pressure evicts the retained conversation the repeat resumes from.
    const ninfer::GenerationResult resumed = run(true, "resumed second turn");
    const ninfer::GenerationResult repeat  = run(true, "repeated second turn");
    const ninfer::GenerationResult cold    = run(false, "cold second turn");

    const std::uint32_t first_tokens = first.prompt.prompt_tokens;
    const auto reused_most           = [&](const ninfer::GenerationResult& result) {
        return result.prefix_reuse_path != ninfer::PrefixReusePath::Root &&
               static_cast<std::uint64_t>(result.reused_prompt_tokens) * 5U >=
                   static_cast<std::uint64_t>(first_tokens) * 4U;
    };
    if (first_tokens < kMinimumLongPrompt) {
        std::cerr << label << ": the first turn has only " << first_tokens << " prompt tokens\n";
        ++failures;
    }
    for (const auto& [result, leg] :
         {std::pair{&resumed, "resumed second turn"}, std::pair{&repeat, "repeated second turn"}}) {
        failures += check_answer(*result, "144", label);
        if (!reused_most(*result)) {
            std::cerr << label << ", " << leg << ": reused " << result->reused_prompt_tokens
                      << " prompt tokens, less than 80 % of the first turn's " << first_tokens
                      << '\n';
            ++failures;
        }
        if (result->speculative.backend != ninfer::SpeculativeBackend::Mtp ||
            result->speculative.rounds == 0) {
            std::cerr << label << ", " << leg << ": no MTP round ran\n";
            ++failures;
        }
        if (result->generated_token_ids != cold.generated_token_ids) {
            std::cerr << label << ", " << leg << ": the answer differs from the cold prefill's (\""
                      << result->content << "\" against \"" << cold.content << "\")\n";
            ++failures;
        }
    }
    failures += check_answer(cold, "144", label);
    if (cold.reused_prompt_tokens != 0 || cold.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << label << ": the cold second turn resumed a retained prefix\n";
        ++failures;
    }
    if (repeat.reused_prompt_tokens == repeat.prompt.prompt_tokens) {
        std::cout << label << ": the repeated second turn resumed its whole prompt\n";
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: tensor parallelism needs two CUDA devices\n";
        return 77;
    }
    const bool optimized = argc > 1 && std::strcmp(argv[1], "optimized") == 0;
    try {
        // One Engine at a time: each holds the whole model across both devices.
        std::vector<ProbeRun> reference;
        {
            ninfer::Engine ordinary(engine_options(artifact, false, false));
            reference = run_probes(ordinary);
        }
        int failures = 0;
        {
            ninfer::Engine engine(engine_options(artifact, true, optimized));
            failures += exercise_parity(engine, reference);
            failures += exercise_concurrent(engine);
            failures += exercise_accepted_lengths(engine);
            failures += exercise_prompt_token_ids(engine);
            failures += exercise_repeated(engine);
        }
        {
            ninfer::Engine engine(long_engine_options(artifact, optimized, 4));
            failures += exercise_prefix_reuse(engine, "MTP prefix reuse, 4 lanes");
        }
        {
            ninfer::Engine engine(long_engine_options(artifact, optimized, 1));
            failures += exercise_prefix_reuse(engine, "MTP prefix reuse, 1 lane");
        }
        if (failures != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 MTP Engine failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
