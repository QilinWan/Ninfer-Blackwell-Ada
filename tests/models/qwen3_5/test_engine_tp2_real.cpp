// Two-device tensor-parallel (tp 2) Engine on NINFER_TEST_ARTIFACT, the public generation route
// ninfer-serve uses: Qwen3.8 27B does not fit one 16 GB device, so the checks are semantic rather
// than a comparison with tp 1. CUDA Graphs, the context cache and two request lanes are on, as in
// the served default.
//
//   * single: "Quanto fa 17*23?" is prefilled and decoded greedily; the answer contains "391".
//   * concurrent: a long answer keeps lane 0 decoding while a second prompt is admitted on lane 1,
//     prefilled and decoded in shared rounds. Rank 1 must prefill each lane through that lane's
//     own KV execution row, so both answers must still be right (a stale rank 1 row writes lane
//     1's half of the KV heads into lane 0's pages). Two short arithmetic prompts then run
//     together as well.
//   * prefix reuse: a second turn repeats the first turn's conversation and asks a new question;
//     it reuses the retained prefix (restored KV pages and StateImages on both ranks) and answers
//     it correctly.
//   * long prefix, shaped like ninfer-serve's Chat Completions traffic (no session key, the
//     default implicit shared-prefix marker after the last message): a ~6k-token document with a
//     question, then the same conversation with the answer and a new question. The second turn
//     must reuse at least 80 % of the first prompt, with 4 lanes and with 1 lane (32k context).
//
// Returns 77 without the artifact or below two CUDA devices.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext     = 8192;
constexpr std::uint32_t kLongMaxContext = 32768;

ninfer::EngineOptions engine_options(const char* artifact) {
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
    return options;
}

// The serving shape: 32k context and KV, `lanes` lanes, tp 2 context-cache defaults.
ninfer::EngineOptions long_engine_options(const char* artifact, std::uint32_t lanes) {
    ninfer::EngineOptions options = engine_options(artifact);
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

int exercise_load(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.devices.size() != 2 || load.devices[0].device == load.devices[1].device ||
        load.devices[0].capacity_bytes == 0 || load.devices[1].capacity_bytes == 0 ||
        load.devices[0].sharded_bytes == 0 || load.devices[1].sharded_bytes == 0) {
        std::cerr << "load summary does not describe two ranks holding weight shards\n";
        return 1;
    }
    const ninfer::EngineOptions& options = engine.options();
    if (options.context_cache.host_state_slots != 0 ||
        options.context_cache.host_kv_capacity_bytes != 0 ||
        options.context_cache.device_state_slots.value_or(0) < 4 ||
        options.context_cache.max_private_continuations.value_or(0) < 8) {
        std::cerr << "tp 2 context-cache defaults were not applied\n";
        return 1;
    }
    return 0;
}

int exercise_single(ninfer::Engine& engine) {
    const ninfer::GenerationResult result = engine.generate(
        engine.prepare(user_prompt("Quanto fa 17*23? Rispondi col solo numero.")), greedy(32));
    return check_answer(result, "391", "single request");
}

int exercise_concurrent(ninfer::Engine& engine) {
    int failures                      = 0;
    const ninfer::RuntimeStats before = engine.runtime_stats();
    // Submitted before either is waited on: the long answer holds lane 0 while the second prompt
    // is admitted on lane 1 and joins its decode rounds.
    ninfer::GenerationHandle counting = engine.submit(
        engine.prepare(user_prompt("Scrivi i numeri interi da 1 a 40, separati da una virgola e "
                                   "uno spazio, senza nient'altro.")),
        greedy(160));
    ninfer::GenerationHandle product = engine.submit(
        engine.prepare(user_prompt("Quanto fa 17*23? Rispondi col solo numero.")), greedy(32));
    const ninfer::GenerationResult counted    = counting.wait();
    const ninfer::GenerationResult multiplied = product.wait();
    const ninfer::RuntimeStats after          = engine.runtime_stats();
    failures += check_answer(counted, "38, 39, 40", "concurrent long lane");
    failures += check_answer(multiplied, "391", "concurrent second lane");
    const std::uint64_t rounds = after.decode_rounds - before.decode_rounds;
    const std::uint64_t rows   = after.decode_row_rounds - before.decode_row_rounds;
    if (rounds == 0 || rows <= rounds) {
        std::cerr << "concurrent leg ran no two-row decode round (" << rounds << " rounds, " << rows
                  << " rows)\n";
        ++failures;
    }

    ninfer::GenerationHandle first = engine.submit(
        engine.prepare(user_prompt("Quanto fa 17*23? Rispondi col solo numero.")), greedy(32));
    ninfer::GenerationHandle second = engine.submit(
        engine.prepare(user_prompt("Quanto fa 12*12? Rispondi col solo numero.")), greedy(32));
    failures += check_answer(first.wait(), "391", "concurrent pair, first");
    failures += check_answer(second.wait(), "144", "concurrent pair, second");
    return failures;
}

int exercise_prefix_reuse(ninfer::Engine& engine) {
    const auto conversation = [] {
        ninfer::PromptInput input = user_prompt(
            "Tieni a mente il codice 4817 e il colore verde. Per ora rispondi soltanto: OK.");
        input.context_cache.session_key = "tp2-real";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation()), greedy(16, true));
    if (first.generated_token_ids.empty()) {
        std::cerr << "prefix reuse: the first turn generated nothing\n";
        return 1;
    }

    ninfer::PromptInput followup = conversation();
    ninfer::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
    followup.messages.push_back(std::move(assistant));
    ninfer::ChatMessage question;
    question.role = ninfer::ChatRole::User;
    question.parts.push_back(
        ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                            .text  = "Quanto fa 17*23? Rispondi col solo numero.",
                            .media = {}});
    followup.messages.push_back(std::move(question));
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(std::move(followup)), greedy(32, true));
    int failures = check_answer(second, "391", "prefix reuse second turn");
    if (second.reused_prompt_tokens == 0 ||
        second.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << "prefix reuse: the second turn reused no prompt tokens (path "
                  << static_cast<int>(second.prefix_reuse_path) << ")\n";
        ++failures;
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
    std::string text = "Leggi il registro del magazzino e poi rispondi alla domanda.\n\n";
    for (int line = 1; line <= 240; ++line) {
        text += "Riga " + std::to_string(line) + ": il magazzino di Verona ha ricevuto " +
                std::to_string(line * 7) + " casse di mele e " + std::to_string(line * 3) +
                " casse di pere, spedite il giorno " + std::to_string(1 + line % 28) + ".\n";
    }
    return text;
}

void print_cache_stats(const ninfer::RuntimeStats& before, const ninfer::RuntimeStats& after) {
    std::cerr << "  captures completed " << after.active_captures_completed << " aborted "
              << after.active_captures_aborted << "; selections root "
              << after.root_selections - before.root_selections << " endpoint "
              << after.private_endpoint_selections - before.private_endpoint_selections
              << " turn-closure "
              << after.private_turn_closure_selections - before.private_turn_closure_selections
              << " anchor "
              << after.private_long_anchor_selections - before.private_long_anchor_selections
              << " shared "
              << after.shared_stable_prefix_selections - before.shared_stable_prefix_selections
              << "; computed prefill "
              << after.computed_prefill_tokens - before.computed_prefill_tokens
              << "; evicted private " << after.pressure_private_owners_evicted << " shared "
              << after.pressure_shared_owners_evicted << " checkpoints dropped "
              << after.pressure_checkpoints_dropped << '\n';
}

int exercise_long_prefix_reuse(ninfer::Engine& engine, const char* label) {
    const std::string document = long_document();
    const auto first_turn      = [&] {
        ninfer::PromptInput input =
            user_prompt(document + "\nDomanda: quanto fa 17*23? Rispondi col solo numero.");
        mark_like_chat_completions(input);
        return input;
    };
    const ninfer::RuntimeStats start = engine.runtime_stats();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_turn()), greedy(32, true));
    int failures = check_answer(first, "391", label);

    ninfer::PromptInput followup = first_turn();
    followup.context_cache       = {};
    ninfer::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = first.content, .media = {}});
    followup.messages.push_back(std::move(assistant));
    ninfer::ChatMessage question;
    question.role = ninfer::ChatRole::User;
    question.parts.push_back(
        ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                            .text  = "Quanto fa 12*12? Rispondi col solo numero.",
                            .media = {}});
    followup.messages.push_back(std::move(question));
    mark_like_chat_completions(followup);
    const ninfer::RuntimeStats middle = engine.runtime_stats();
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(std::move(followup)), greedy(32, true));
    const ninfer::RuntimeStats end = engine.runtime_stats();
    failures += check_answer(second, "144", label);
    const std::uint32_t first_tokens = first.prompt.prompt_tokens;
    if (first_tokens < 4096 || static_cast<std::uint64_t>(second.reused_prompt_tokens) * 5U <
                                   static_cast<std::uint64_t>(first_tokens) * 4U) {
        std::cerr << label << ": second turn reused " << second.reused_prompt_tokens << " of "
                  << second.prompt.prompt_tokens << " prompt tokens (first turn " << first_tokens
                  << ", path " << static_cast<int>(second.prefix_reuse_path) << ")\n";
        std::cerr << "  after the first turn:";
        print_cache_stats(start, middle);
        std::cerr << "  after the second turn:";
        print_cache_stats(middle, end);
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
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
    try {
        ninfer::Engine engine(engine_options(artifact));
        int failures = exercise_load(engine);
        failures += exercise_single(engine);
        failures += exercise_concurrent(engine);
        failures += exercise_prefix_reuse(engine);
        if (failures != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 Engine failed: " << error.what() << '\n';
        return 1;
    }
    try {
        int failures = 0;
        {
            ninfer::Engine engine(long_engine_options(artifact, 4));
            failures += exercise_long_prefix_reuse(engine, "long prefix, 4 lanes");
        }
        {
            ninfer::Engine engine(long_engine_options(artifact, 1));
            failures += exercise_long_prefix_reuse(engine, "long prefix, 1 lane");
        }
        if (failures != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 Engine failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
