// Two-device tensor-parallel (tp 2) Engine with DFlash2 speculative decoding (K=4, optimized
// proposal head) on NINFER_TEST_ARTIFACT, through the public generation route ninfer-serve uses.
// Qwen3.8 27B does not fit one 16 GB device, so the reference is the same model at tp 2 without
// speculation. CUDA Graphs, the context cache and two request lanes are on, as in the served
// default.
//
//   * parity: three short greedy prompts run first on a tp 2 Engine without speculation and then
//     on the DFlash2 Engine. Greedy verification is lossless with respect to the target's own
//     argmax, and these answers are short enough that the two schedules produce the same tokens,
//     so the token ids must be identical; the checkable answers must also be right. A drafter fed
//     broken features, a draft copy that reaches rank 1 late, or a rank 1 verification that
//     disagrees with rank 0 (KV rows, GDN records, masked columns) changes the answer.
//   * acceptance: over the parity prompts at least one drafted token is accepted and every
//     request runs DFlash2 rounds. Rank 0's drafter reads the target features its tap captures
//     from the replicated residual; features captured from a partial residual collapse the
//     acceptance while the output stays right.
//   * concurrent: "Quanto fa 17*23?" answers 391 while a long answer holds the other lane, in
//     shared two-row DFlash2 rounds.
//   * prefix reuse: a second turn repeats the first turn's conversation and asks a new question.
//     It resumes the retained prefix (restored KV pages and StateImages on both ranks, the
//     drafter's rings on rank 0): the second turn must reuse prompt tokens and answer correctly.
//
// Returns 77 without the artifact or below two CUDA devices.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext  = 8192;
constexpr std::uint32_t kDraftTokens = 4;

ninfer::EngineOptions engine_options(const char* artifact, bool dflash2) {
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
    if (dflash2) {
        options.speculative.backend       = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens  = kDraftTokens;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
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

int check_dflash2(const ninfer::GenerationResult& result, const char* label) {
    if (result.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
        result.speculative.rounds != 0) {
        return 0;
    }
    std::cerr << label << ": no DFlash2 round ran\n";
    return 1;
}

struct Probe {
    const char* label;
    const char* prompt;
    const char* expected;
};

const std::vector<Probe>& probes() {
    static const std::vector<Probe> list{
        {"product", "Quanto fa 17*23? Rispondi col solo numero.", "391"},
        {"square", "Quanto fa 12*12? Rispondi col solo numero.", "144"},
        {"capital", "Qual e' la capitale d'Italia? Rispondi con una sola parola.", "Roma"},
    };
    return list;
}

constexpr std::uint32_t kProbeTokens = 32;

std::vector<ninfer::GenerationResult> run_probes(ninfer::Engine& engine) {
    std::vector<ninfer::GenerationResult> results;
    for (const Probe& probe : probes()) {
        results.push_back(
            engine.generate(engine.prepare(user_prompt(probe.prompt)), greedy(kProbeTokens)));
    }
    return results;
}

int exercise_parity(ninfer::Engine& engine,
                    const std::vector<ninfer::GenerationResult>& reference) {
    int failures           = 0;
    std::uint64_t drafted  = 0;
    std::uint64_t accepted = 0;
    const auto& probe_list = probes();
    const auto results     = run_probes(engine);
    for (std::size_t index = 0; index < probe_list.size(); ++index) {
        const Probe& probe                     = probe_list[index];
        const ninfer::GenerationResult& result = results[index];
        const ninfer::GenerationResult& base   = reference[index];
        drafted += result.speculative.drafted_tokens;
        accepted += result.speculative.accepted_tokens;
        const bool same = result.generated_token_ids == base.generated_token_ids;
        std::cout << "parity " << probe.label << ": " << (same ? "identical" : "diverged")
                  << ", acceptance " << result.speculative.accepted_tokens << "/"
                  << result.speculative.drafted_tokens << '\n';
        failures += check_dflash2(result, probe.label);
        if (!same) {
            std::cerr << "parity " << probe.label
                      << ": DFlash2 and ordinary tp 2 answers differ\n  DFlash2:  \""
                      << result.content << "\"\n  ordinary: \"" << base.content << "\"\n";
            ++failures;
        }
        failures += check_answer(result, probe.expected, probe.label);
        if (!contains(base.content, probe.expected)) {
            std::cerr << "parity " << probe.label << ": the ordinary reference is wrong\n";
            ++failures;
        }
    }
    std::cout << "DFlash2 acceptance over the parity prompts: " << accepted << "/" << drafted
              << '\n';
    if (accepted == 0) {
        std::cerr << "DFlash2 accepted no drafted token over the parity prompts\n";
        ++failures;
    }
    return failures;
}

int exercise_concurrent(ninfer::Engine& engine) {
    int failures                      = 0;
    const ninfer::RuntimeStats before = engine.runtime_stats();
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
    failures += check_dflash2(counted, "concurrent long lane");
    failures += check_dflash2(multiplied, "concurrent second lane");
    const std::uint64_t rounds = after.decode_rounds - before.decode_rounds;
    const std::uint64_t rows   = after.decode_row_rounds - before.decode_row_rounds;
    if (rounds == 0 || rows <= rounds) {
        std::cerr << "concurrent leg ran no two-row DFlash2 round (" << rounds << " rounds, "
                  << rows << " rows)\n";
        ++failures;
    }
    if (counted.speculative.accepted_tokens == 0) {
        std::cerr << "concurrent long lane accepted no draft\n";
        ++failures;
    }
    return failures;
}

int exercise_prefix_reuse(ninfer::Engine& engine) {
    const auto conversation = [] {
        ninfer::PromptInput input = user_prompt(
            "Tieni a mente il codice 4817 e il colore verde. Per ora rispondi soltanto: OK.");
        input.context_cache.session_key = "dflash2-tp2-real";
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
    failures += check_dflash2(second, "prefix reuse second turn");
    std::cout << "prefix reuse: second turn reused " << second.reused_prompt_tokens
              << " prompt tokens\n";
    if (second.reused_prompt_tokens == 0 ||
        second.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << "prefix reuse: the second turn reused no prompt tokens (path "
                  << static_cast<int>(second.prefix_reuse_path) << ")\n";
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
        // One Engine at a time: each holds the whole model across both devices.
        std::vector<ninfer::GenerationResult> reference;
        {
            ninfer::Engine ordinary(engine_options(artifact, false));
            reference = run_probes(ordinary);
        }
        ninfer::Engine engine(engine_options(artifact, true));
        int failures = exercise_parity(engine, reference);
        failures += exercise_concurrent(engine);
        failures += exercise_prefix_reuse(engine);
        if (failures != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 DFlash2 Engine failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
