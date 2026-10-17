// Two-device tensor-parallel (tp 2) Engine with a resident Vision tower on NINFER_TEST_ARTIFACT,
// through the public generation route ninfer-serve uses. This tree holds the tower whole on rank 0
// (--vision-device must name the first --devices entry) and sizes rank 0's arena with the encode
// workspace alone; it has no rank 1 receiver for the merged embeddings (that is the deferred layout
// batch), so the two-device schedule rejects a prompt that carries media. Qwen3.8 27B does not fit
// one 16 GB device, so every reference is the same model at tp 2. The images are synthetic
// solid-color PPMs; the per-item Vision extent is 256 tokens (--max-vision-tokens).
//
// Default mode:
//   * rejections before the artifact is read: a vision_device outside devices, a tower on rank 1 at
//     tp 2, a vision_device or max_vision_tokens without Vision, and a max_vision_tokens below 64.
//   * text invariance: a text prompt answers with the same token ids on the tp 2 Engine without
//     Vision and on the Vision Engine, whose arena carries the extra Vision phase.
//   * media: with the tower resident on rank 0, each image prompt is refused by the two-device
//     schedule (execution/text.cpp: multimodal prefill) and produces no answer; the text prompt
//     still answers correctly afterwards.
// `mtp` and `dflash2` modes run a text conversation with MTP (K=3) or DFlash2 (K=4, optimized
// proposal head) on the Vision Engine: speculative rounds must run, and a second turn must resume
// its retained prefix through the bridge.
//
// Returns 77 without the artifact or below two CUDA devices.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext       = 8192;
constexpr std::uint32_t kMaxVisionTokens  = 256;
constexpr int kImageEdge                  = 768;
constexpr std::uint32_t kAnswerTokens     = 16;
constexpr std::uint32_t kTemplateTokenCap = 64;

enum class Mode { Parity, Mtp, DFlash2 };

ninfer::EngineOptions engine_options(const char* artifact, bool vision,
                                     std::optional<int> vision_device, Mode mode) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.tp                   = 2;
    options.device               = 0;
    options.devices              = {0, 1};
    options.max_context          = kMaxContext;
    // One lane: the KV pool holds exactly one max_context sequence.
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk        = 1024;
    options.max_concurrency      = 1;
    options.max_pending_requests = 2;
    // Rank 1's KV and StateImages have no Host tier: this tree's Engine requires the single
    // consolidated Host capacity to be 0 at --tp 2.
    options.context_cache.host_capacity_bytes = 0;
    if (vision) {
        options.enable_vision     = true;
        options.vision_device     = vision_device;
        options.max_vision_tokens = kMaxVisionTokens;
    }
    if (mode == Mode::Mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    } else if (mode == Mode::DFlash2) {
        options.speculative.backend       = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens  = 4;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    return options;
}

std::vector<std::uint8_t> solid_ppm(std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    const std::string header =
        "P6\n" + std::to_string(kImageEdge) + ' ' + std::to_string(kImageEdge) + "\n255\n";
    std::vector<std::uint8_t> ppm(header.begin(), header.end());
    ppm.reserve(ppm.size() + static_cast<std::size_t>(kImageEdge) * kImageEdge * 3);
    for (int pixel = 0; pixel < kImageEdge * kImageEdge; ++pixel) {
        ppm.push_back(red);
        ppm.push_back(green);
        ppm.push_back(blue);
    }
    return ppm;
}

ninfer::MessagePart text_part(std::string text) {
    return ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}};
}

ninfer::PromptInput text_prompt(std::string text) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(text_part(std::move(text)));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::PromptInput image_prompt(const std::vector<std::uint8_t>& image, std::string name) {
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = ninfer::MediaKind::Image;
    media.media.bytes       = image;
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = std::move(name);
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(media));
    message.parts.push_back(text_part("What color is this image? Answer with one word."));
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

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

int check_answer(const ninfer::GenerationResult& result, std::string_view expected,
                 const std::string& label) {
    if (lowercase(result.content).find(expected) != std::string::npos) { return 0; }
    std::cerr << label << ": expected \"" << expected << "\" in the answer, got \""
              << result.content << "\" (" << result.generated_token_ids.size() << " tokens)\n";
    return 1;
}

struct ImageProbe {
    const char* label;
    std::vector<std::uint8_t> image;
    const char* expected;
};

std::vector<ImageProbe> image_probes() {
    std::vector<ImageProbe> probes;
    probes.push_back({"red image", solid_ppm(220, 20, 20), "red"});
    probes.push_back({"blue image", solid_ppm(20, 40, 220), "blue"});
    return probes;
}

constexpr const char* kTextPrompt = "What is 17*23? Answer with the number only.";

// The image probes, each refused by the two-device schedule: this tree has no rank 1 receiver for
// the merged embeddings, so the prefill entry rejects a prompt that carries media.
int expect_media_rejected(ninfer::Engine& engine, const std::string& engine_label) {
    int failures = 0;
    for (const ImageProbe& probe : image_probes()) {
        const std::string label = engine_label + " " + probe.label;
        bool refused            = false;
        try {
            const ninfer::GenerationResult result = engine.generate(
                engine.prepare(image_prompt(probe.image, "probe.ppm")), greedy(kAnswerTokens));
            std::cerr << label << ": a media prompt ran at tp 2 and answered \"" << result.content
                      << "\"\n";
            ++failures;
        } catch (const std::exception& error) {
            refused            = true;
            const std::string what(error.what());
            std::cout << label << ": refused with \"" << what << "\"\n";
            if (what.find("multimodal prefill") == std::string::npos &&
                what.find("tensor-parallel width 2") == std::string::npos) {
                std::cerr << label << ": refused by something other than the split schedule\n";
                ++failures;
            }
        }
        if (!refused) { ++failures; }
    }
    return failures;
}

int compare_ids(const ninfer::GenerationResult& actual, const ninfer::GenerationResult& expected,
                const std::string& label) {
    if (actual.generated_token_ids == expected.generated_token_ids) {
        std::cout << label << ": identical\n";
        return 0;
    }
    std::cerr << label << ": token ids differ\n  got:      \"" << actual.content
              << "\"\n  expected: \"" << expected.content << "\"\n";
    return 1;
}

bool rejected(const std::function<void(ninfer::EngineOptions&)>& edit, const char* artifact) {
    ninfer::EngineOptions options = engine_options(artifact, true, std::nullopt, Mode::Parity);
    edit(options);
    try {
        ninfer::Engine engine(std::move(options));
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int exercise_rejections(const char* artifact) {
    int failures      = 0;
    const auto expect = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    expect(rejected([](ninfer::EngineOptions& o) { o.vision_device = 2; }, artifact),
           "a vision_device outside devices was accepted");
    // This tree encodes on rank 0 alone (planning/startup.cpp:validate_target_options,
    // runtime/engine/model_instance.cpp): the tower needs the rank whose arena carries its encode
    // phase and whose Parameters hold the tower.
    expect(rejected([](ninfer::EngineOptions& o) { o.vision_device = 1; }, artifact),
           "a Vision tower on rank 1 was accepted at tp 2");
    expect(rejected(
               [](ninfer::EngineOptions& o) {
                   o.enable_vision     = false;
                   o.max_vision_tokens = std::nullopt;
                   o.vision_device     = 1;
               },
               artifact),
           "a vision_device without Vision was accepted");
    expect(rejected([](ninfer::EngineOptions& o) { o.max_vision_tokens = 32; }, artifact),
           "a max_vision_tokens below 64 was accepted");
    return failures;
}

int exercise_parity(const char* artifact) {
    int failures = exercise_rejections(artifact);

    ninfer::GenerationResult text_reference;
    {
        ninfer::Engine plain(engine_options(artifact, false, std::nullopt, Mode::Parity));
        text_reference = plain.generate(plain.prepare(text_prompt(kTextPrompt)), greedy(32));
    }
    failures += check_answer(text_reference, "391", "text without Vision");

    ninfer::GenerationResult rank0_text;
    {
        ninfer::Engine engine(engine_options(artifact, true, 0, Mode::Parity));
        const std::uint32_t tokens =
            engine.count_tokens(image_prompt(image_probes().front().image, "count.ppm"));
        std::cout << "image prompt tokens at --max-vision-tokens " << kMaxVisionTokens << ": "
                  << tokens << '\n';
        if (tokens > kMaxVisionTokens + kTemplateTokenCap) {
            std::cerr << "a " << kImageEdge << "x" << kImageEdge
                      << " image was not resized to the per-item Vision extent\n";
            ++failures;
        }
        // A refused media request ends the Engine (the same failure route a decode round takes),
        // so the text prompt that must answer runs first: it is what shows that a resident tower
        // and its rank-0 encode phase change nothing about the text schedule.
        rank0_text = engine.generate(engine.prepare(text_prompt(kTextPrompt)), greedy(32));
        failures += compare_ids(rank0_text, text_reference, "text with Vision vs without");
        failures += expect_media_rejected(engine, "tower on rank 0");
    }
    return failures;
}

int exercise_speculative(const char* artifact, Mode mode) {
    const bool mtp = mode == Mode::Mtp;
    const std::string label(mtp ? "MTP" : "DFlash2");
    // The tower sits on rank 0, the rank that may encode (the admission above rejects any other
    // placement), and the prompts are text: a media prompt has no two-device schedule here.
    ninfer::Engine engine(engine_options(artifact, true, 0, mode));
    const ninfer::SpeculativeBackend backend =
        mtp ? ninfer::SpeculativeBackend::Mtp : ninfer::SpeculativeBackend::DFlash2;
    int failures = 0;
    const ninfer::GenerationResult answer =
        engine.generate(engine.prepare(text_prompt(kTextPrompt)), greedy(32));
    std::cout << label << " answer: \"" << answer.content << "\" in " << answer.timings.total_seconds
              << " s\n";
    failures += check_answer(answer, "391", label + " text answer");
    if (answer.speculative.backend != backend || answer.speculative.rounds == 0) {
        std::cerr << label << ": the answer ran no speculative round (backend "
                  << static_cast<int>(answer.speculative.backend) << ", rounds "
                  << answer.speculative.rounds << ")\n";
        ++failures;
    }

    // A second turn of the same conversation resumes its retained prefix (the MTP bridge under
    // MTP, the drafter's rings under DFlash2) on the two-device schedule.
    constexpr const char* kFirstTurn = "What is 17*23? Answer with the number only.";
    const auto conversation = [] {
        ninfer::PromptInput input       = text_prompt(kFirstTurn);
        input.context_cache.session_key = "vision-tp2-real";
        // This tree has no per-request cache-retention knob: the session lineage is advanced by
        // the default update_session_index, which is what keeps the retained prefix resolvable.
        return input;
    };
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation()), greedy(kAnswerTokens, true));
    failures += check_answer(first, "391", label + " first turn");
    ninfer::PromptInput followup = conversation();
    ninfer::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.parts.push_back(text_part(first.content));
    followup.messages.push_back(std::move(assistant));
    ninfer::ChatMessage question;
    question.role = ninfer::ChatRole::User;
    question.parts.push_back(text_part("Write that number again, digits only."));
    followup.messages.push_back(std::move(question));
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(std::move(followup)), greedy(kAnswerTokens, true));
    std::cout << label << " second turn: \"" << second.content << "\", reused "
              << second.reused_prompt_tokens << " prompt tokens\n";
    failures += check_answer(second, "391", label + " second turn");
    if (second.reused_prompt_tokens == 0 ||
        second.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << label << " second turn reused no prompt tokens\n";
        ++failures;
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
    Mode mode = Mode::Parity;
    if (argc > 1) {
        const std::string_view selected(argv[1]);
        if (selected == "mtp") {
            mode = Mode::Mtp;
        } else if (selected == "dflash2") {
            mode = Mode::DFlash2;
        } else {
            std::cerr << "usage: " << argv[0] << " [mtp|dflash2]\n";
            return 2;
        }
    }
    try {
        // One Engine at a time: each holds the whole model across both devices.
        const int failures =
            mode == Mode::Parity ? exercise_parity(artifact) : exercise_speculative(artifact, mode);
        if (failures != 0) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "tp 2 Vision Engine failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
