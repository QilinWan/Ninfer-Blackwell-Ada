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
//   * single and concurrent: "Quanto fa 17*23?" answers 391 alone and while a long answer holds
//     the other lane, in shared two-row MTP rounds; a second pair runs together as well.
//   * repeated requests: more than the private continuation catalog's worth of requests in series,
//     each binding and releasing both ranks' MTP KV rows (a leaked rank 1 row fails the bind of a
//     later request).
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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext        = 8192;
constexpr std::uint32_t kDraftTokens       = 3;
constexpr std::size_t kMinimumCommonPrefix = 24;
constexpr double kMinimumAcceptance        = 0.25;
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
    // Rank 1's KV and StateImages have no Host copy, so the Host budget must be zero; the Device
    // checkpoint pool floor comes from the Engine's tp 2 default (max(2 * concurrency, 8)).
    options.context_cache.host_capacity_bytes = 0;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = kDraftTokens;
        options.speculative.proposal_head =
            optimized ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
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

ninfer::RequestOptions greedy(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
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
        {"product", "Quanto fa 17*23? Rispondi col solo numero.", 32, "391"},
        {"square", "Quanto fa 12*12? Rispondi col solo numero.", 32, "144"},
        {"code",
         "Scrivi una funzione Python iterativa che restituisce il fattoriale di n. Rispondi solo "
         "con il codice.",
         128, "def "},
        {"text",
         "Scrivi un paragrafo di circa 150 parole sulla fondazione di Roma, senza titoli ne' "
         "elenchi.",
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
        std::cerr << "concurrent leg ran no two-row MTP round (" << rounds << " rounds, " << rows
                  << " rows)\n";
        ++failures;
    }
    if (counted.speculative.accepted_tokens == 0) {
        std::cerr << "concurrent long lane accepted no draft\n";
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

int exercise_repeated(ninfer::Engine& engine) {
    int failures = 0;
    for (std::uint32_t index = 0; index < kRepeatedRequests; ++index) {
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare(user_prompt("Quanto fa " + std::to_string(11 + index) +
                                                       "*3? Rispondi col solo numero.")),
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
        ninfer::Engine engine(engine_options(artifact, true, optimized));
        int failures = exercise_parity(engine, reference);
        failures += exercise_concurrent(engine);
        failures += exercise_repeated(engine);
        if (failures != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 MTP Engine failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
