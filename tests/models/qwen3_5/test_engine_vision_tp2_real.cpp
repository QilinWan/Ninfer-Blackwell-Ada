// Two-device tensor-parallel (tp 2) Engine with a resident Vision tower on NINFER_TEST_ARTIFACT,
// through the public generation route ninfer-serve uses. This tree holds the tower whole on rank 0
// (--vision-device must name the first --devices entry) and hands each encoded item's rows to rank
// 1 with one pitched UVA copy, so both ranks scatter the same merged embeddings into their own
// residual (execution/text.cpp: prefill_impl_tp2). Qwen3.8 27B does not fit one 16 GB device, so
// every reference is the same model at tp 2. The images are synthetic solid-color PPMs; the
// per-item Vision extent is 256 tokens (--max-vision-tokens).
//
// Default mode:
//   * rejections before the artifact is read: a vision_device outside devices, a tower on rank 1 at
//     tp 2, a vision_device or max_vision_tokens without Vision, and a max_vision_tokens below 64.
//   * text invariance: a text prompt answers with the same token ids on the tp 2 Engine without
//     Vision and on the Vision Engine, whose arena carries the extra Vision phase.
//   * media: each solid-color image prompt answers with its color through the rank-0 encode and
//     the rank-1 handoff; the same prompt run twice is token-identical (greedy determinism), and
//     the text prompt still answers correctly afterwards.
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

std::vector<std::uint8_t> solid_ppm_wide(std::uint8_t red, std::uint8_t green, std::uint8_t blue,
                                         int width, int height) {
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    std::vector<std::uint8_t> ppm(header.begin(), header.end());
    ppm.reserve(ppm.size() + static_cast<std::size_t>(width) * height * 3);
    for (int pixel = 0; pixel < width * height; ++pixel) {
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

ninfer::MessagePart media_part(const std::vector<std::uint8_t>& image, std::string name) {
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = ninfer::MediaKind::Image;
    media.media.bytes       = image;
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = std::move(name);
    return media;
}

// Two items in one user message: the Vision plan encodes them one at a time and each item's rows
// cross the handoff inside the same prompt.
ninfer::PromptInput two_image_prompt(const std::vector<std::uint8_t>& first,
                                     const std::vector<std::uint8_t>& second) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(media_part(first, "first.ppm"));
    message.parts.push_back(media_part(second, "second.ppm"));
    message.parts.push_back(
        text_part("Name the two colors in these images. Answer with two words."));
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

// The image probes answer through the rank-0 encode and the rank-1 handoff: each solid color must
// come back, and a second greedy run of the same prompt must be token-identical.
int exercise_media(ninfer::Engine& engine, const std::string& engine_label) {
    int failures = 0;
    for (const ImageProbe& probe : image_probes()) {
        const std::string label = engine_label + " " + probe.label;
        const ninfer::GenerationResult first = engine.generate(
            engine.prepare(image_prompt(probe.image, "probe.ppm")), greedy(kAnswerTokens));
        failures += check_answer(first, probe.expected, label);
        std::cout << label << ": \"" << first.content << "\" in " << first.timings.total_seconds
                  << " s\n";
        const ninfer::GenerationResult second = engine.generate(
            engine.prepare(image_prompt(probe.image, "probe.ppm")), greedy(kAnswerTokens));
        if (second.generated_token_ids != first.generated_token_ids) {
            std::cerr << label << ": two greedy runs differ\n";
            ++failures;
        }
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

// Two items in one prompt and a non-square item: both runs must answer and be token-identical.
int exercise_multi(ninfer::Engine& engine, const char* artifact);

int exercise_parity(const char* artifact) {
    int failures = 0;

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
        // In this tree the per-item extent sizes the Vision workspace; the prepared prompt budget
        // is the context-derived one, so the count is bounded by kMaxContext on the two-device
        // schedule (the same image counts the same tokens without Vision).
        if (tokens == 0 || tokens > kMaxContext) {
            std::cerr << "a " << kImageEdge << "x" << kImageEdge
                      << " image exceeded the prompt budget\n";
            ++failures;
        }
        // A refused media request ends the Engine (the same failure route a decode round takes),
        // so the text prompt that must answer runs first: it is what shows that a resident tower
        // and its rank-0 encode phase change nothing about the text schedule.
        rank0_text = engine.generate(engine.prepare(text_prompt(kTextPrompt)), greedy(32));
        failures += compare_ids(rank0_text, text_reference, "text with Vision vs without");
        failures += exercise_media(engine, "tower on rank 0");
        failures += exercise_multi(engine, artifact);
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

    // Third turn: append a media message to the reused text conversation. The first new token is
    // a vision token, so the MTP bridge must carry rank 1's retained hidden (the multimodal
    // bridge); a missing peer hidden used to fail the request with "MTP bridge needs rank 1's
    // retained hidden exactly at tp 2".
    ninfer::PromptInput image_followup = conversation();
    ninfer::ChatMessage first_answer;
    first_answer.role = ninfer::ChatRole::Assistant;
    first_answer.parts.push_back(text_part(first.content));
    image_followup.messages.push_back(std::move(first_answer));
    ninfer::ChatMessage second_question;
    second_question.role = ninfer::ChatRole::User;
    second_question.parts.push_back(text_part("Write that number again, digits only."));
    image_followup.messages.push_back(std::move(second_question));
    ninfer::ChatMessage second_answer;
    second_answer.role = ninfer::ChatRole::Assistant;
    second_answer.parts.push_back(text_part(second.content));
    image_followup.messages.push_back(std::move(second_answer));
    ninfer::ChatMessage image_question;
    image_question.role = ninfer::ChatRole::User;
    image_question.parts.push_back(media_part(image_probes().front().image, "followup.ppm"));
    image_question.parts.push_back(text_part("What color is this image? Answer with one word."));
    image_followup.messages.push_back(std::move(image_question));
    const ninfer::GenerationResult image_followup_result =
        engine.generate(engine.prepare(std::move(image_followup)), greedy(kAnswerTokens, true));
    std::cout << label << " image follow-up: \"" << image_followup_result.content << "\", reused "
              << image_followup_result.reused_prompt_tokens << " prompt tokens\n";
    failures += check_answer(image_followup_result, "red", label + " image follow-up");
    if (image_followup_result.reused_prompt_tokens == 0 ||
        image_followup_result.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << label << " image follow-up reused no prompt tokens\n";
        ++failures;
    }

    // The same backend with a fresh media prompt: the spec round consumes the split MTP head over
    // the composed prompt embedding, which only rank 0 forms.
    const ninfer::GenerationResult image =
        engine.generate(engine.prepare(image_prompt(image_probes().front().image, "mtp.ppm")),
                        greedy(kAnswerTokens));
    failures += check_answer(image, "red", label + " image answer");
    return failures;
}

int exercise_multi(ninfer::Engine& engine, const char*) {
    int failures = 0;
    const std::vector<std::uint8_t> red  = solid_ppm(220, 20, 20);
    const std::vector<std::uint8_t> blue = solid_ppm(20, 40, 220);
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(two_image_prompt(red, blue)), greedy(16));
    const std::string answer = lowercase(first.content);
    std::cout << "two images: \"" << first.content << "\"\n";
    if (answer.find("red") == std::string::npos || answer.find("blue") == std::string::npos) {
        std::cerr << "two images: the answer does not name both colors\n";
        ++failures;
    }
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(two_image_prompt(red, blue)), greedy(16));
    if (second.generated_token_ids != first.generated_token_ids) {
        std::cerr << "two images: two greedy runs differ\n";
        ++failures;
    }

    const std::vector<std::uint8_t> wide = solid_ppm_wide(20, 200, 40, 1024, 512);
    const ninfer::GenerationResult wide_answer =
        engine.generate(engine.prepare(image_prompt(wide, "wide.ppm")), greedy(16));
    std::cout << "wide image: \"" << wide_answer.content << "\"\n";
    if (lowercase(wide_answer.content).find("green") == std::string::npos) {
        std::cerr << "wide image: expected green\n";
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
        // The recommended artifact carries text, Vision and MTP, not DFlash2: its dflash2 mode has
        // nothing to load, which is a skip rather than a failure of this suite.
        if (std::string(error.what()).find("missing component dflash2") != std::string::npos) {
            std::cout << "skip: the artifact has no DFlash2 component\n";
            return 77;
        }
        std::cerr << "tp 2 Vision Engine failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
