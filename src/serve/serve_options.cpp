#include "serve/serve_options.h"
#include "product/speculative_options.h"
#include "runtime/contract/yarn.h"
#include "product/tensor_parallel_options.h"
#include "product/vision_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::size_t parse_host_context_mib(const char* text) {
    constexpr std::size_t bytes_per_mib = 1ULL << 20;
    constexpr std::size_t maximum       = std::numeric_limits<std::size_t>::max();
    const std::string_view value(text);
    const std::size_t point      = value.find('.');
    const std::string_view whole = value.substr(0, point);
    std::string_view fraction =
        point == std::string_view::npos ? std::string_view{} : value.substr(point + 1);
    if (whole.empty() || whole.find_first_not_of("0123456789") != std::string_view::npos ||
        (point != std::string_view::npos &&
         (fraction.empty() ||
          fraction.find_first_not_of("0123456789") != std::string_view::npos))) {
        throw std::invalid_argument("--host-context-mib requires a nonnegative decimal MiB value");
    }

    std::size_t whole_mib = 0;
    for (const char character : whole) {
        const std::size_t digit = static_cast<std::size_t>(character - '0');
        if (whole_mib > (maximum / bytes_per_mib - digit) / 10) {
            throw std::invalid_argument("--host-context-mib is out of range");
        }
        whole_mib = whole_mib * 10 + digit;
    }

    // A whole-byte MiB value has at most 20 fractional decimal digits (1 MiB = 2^20 bytes).
    // Use exact integer arithmetic so decimal parsing cannot round the requested budget upward.
    while (!fraction.empty() && fraction.back() == '0') { fraction.remove_suffix(1); }
    if (fraction.size() > 20) {
        throw std::invalid_argument("--host-context-mib must resolve to a whole number of bytes");
    }
    unsigned __int128 numerator = 0;
    unsigned __int128 divisor   = 1;
    for (const char character : fraction) {
        numerator = numerator * 10 + static_cast<unsigned int>(character - '0');
        divisor *= 10;
    }
    numerator *= bytes_per_mib;
    if (numerator % divisor != 0) {
        throw std::invalid_argument("--host-context-mib must resolve to a whole number of bytes");
    }
    const std::size_t fractional_bytes = static_cast<std::size_t>(numerator / divisor);
    const std::size_t whole_bytes      = whole_mib * bytes_per_mib;
    if (fractional_bytes > maximum - whole_bytes) {
        throw std::invalid_argument("--host-context-mib is out of range");
    }
    return whole_bytes + fractional_bytes;
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    if (value == "rk4v4-e8") { return KvCacheStorage::RK4V4E8; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [--host H] [--port N] [--api-key KEY] "
           "[--model-id ID] [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] "
           "[--prefill-chunk N] [--log-stats-interval-ms N] [--device N] "
           "[--tp 1|2 --devices A,B] "
           "[--context-cost-presets FILE] "
           "[--max-request-mib N] [--media-cache-mib N] [--media-live-mib N] "
           "[--media-preprocess-threads N] "
           "[--device-state-slots N] [--host-context-mib N] [--turn-anchors N] "
           "[--log-speculation-detail [--log-speculation-detail-max-steps N]] "
           "[--request-log-jsonl FILE] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|rk4v4-e8] "
           "[--spec mtp|dflash|dflash2 --draft-tokens N] "
           "[--default-max-tokens N] [--default-thinking-budget N] "
           "[--rope-yarn-factor F] [--rope-original-max-position N] "
           "[--vision] [--vision-device N] [--max-vision-tokens N] [--no-cuda-graph] "
           "[--no-tp-mailbox] [--embedding-host] [--no-prefix-reuse] "
           "[--chat-template FILE] [--lm-head-draft] [--no-thinking] [--preserve-thinking] "
           "[--cors] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       --default-max-tokens defaults to " +
           std::to_string(kDefaultMaxTokens) +
           " when omitted\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --media-cache-mib defaults to 1024; 0 disables retained media reuse\n"
           "       --media-live-mib defaults to 2048 and bounds all live BF16 patch payloads\n"
           "       --media-preprocess-threads defaults to 0 (auto, at most 16 workers)\n"
           "       --request-log-jsonl appends full-precision server/request records\n"
           "       --log-speculation-detail adds the per-step accepted draft lengths and the "
           "output/prompt 4-gram overlap to every request_done record (requires "
           "--request-log-jsonl; off by default);\n"
           "       --log-speculation-detail-max-steps caps them per request (default " +
           std::to_string(kDefaultSpeculationDetailMaxSteps) + ", at most " +
           std::to_string(kMaximumSpeculationDetailMaxSteps) +
           "; longer requests are marked truncated)\n"
           "       --model-id overrides the artifact metadata.name reported by the server\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
           "       --vision-device selects the CUDA device that holds the Vision tower and encodes "
           "(default: rank 0's); it must equal --device on one GPU and be one of --devices with "
           "--tp 2, where the other GPU only receives the encoded embeddings\n"
           "       --max-vision-tokens caps the merged Vision tokens of one image or video item "
           "(64-16384, default 16384): larger media are resized and the encode workspace is "
           "planned for N\n"
           "       --kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom; --vram-headroom-mib N changes that margin (requires auto)\n"
           "       --no-prefix-reuse disables cross-request history; request pause/replay "
           "resources remain available\n"
           "       context defaults: device-state=max-concurrency; Host budget is resolved from "
           "8192 MiB plus eight native StateImages\n"
           "       --embedding-host keeps the token embedding table in one copy in pinned host "
           "memory that every GPU reads in place over PCIe, instead of one copy per GPU: it frees "
           "the table's VRAM (1.18 GiB per GPU on Qwen3.8-27B) for the KV pool, identical results\n"
           "       --device-state-slots is extra capacity beyond active lanes; "
           "--host-context-mib bounds shared Host State/KV and in-flight storage in MiB\n"
           "       --tp 2 --devices A,B splits the dense model across two GPUs (rank 0 on A) for "
           "ordinary, --spec mtp or --spec dflash2 --lm-head-draft decoding with bf16 or int8 "
           "KV; it defaults the Device checkpoint pool to max(2x concurrency,8) and the Host "
           "budget to 0\n"
           "       --no-tp-mailbox keeps the captured --tp 2 all-reduces on cross-device copies\n"
           "       --default-thinking-budget caps model-origin thinking for enabled requests; "
           "control tokens count toward the request output limit\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       sampler defaults come from the loaded model and resolved thinking mode; "
           "server flags and request fields override individual values.\n"
           "       --greedy forces temperature 0 (exact argmax).\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;

    bool device_explicit             = false;
    bool host_capacity_explicit      = false;
    bool detail_steps_explicit           = false;
    std::optional<std::size_t> vram_headroom_mib;

    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--vram-headroom-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--vram-headroom-mib"), "vram-headroom-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--vram-headroom-mib is out of range");
            }
            vram_headroom_mib = static_cast<std::size_t>(mib);
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
        } else if (arg == "--host-context-mib") {
            const std::size_t bytes =
                parse_host_context_mib(require_value("--host-context-mib"));
            options.context_cache.host_capacity_bytes = bytes;
            host_capacity_explicit                    = bytes != 0;
        } else if (arg == "--log-speculation-detail") {
            options.log_speculation_detail = true;
        } else if (arg == "--log-speculation-detail-max-steps") {
            const std::uint64_t steps = parse_u64(
                require_value("--log-speculation-detail-max-steps"),
                "log-speculation-detail-max-steps");
            if (steps > (1ULL << 20)) {
                throw std::invalid_argument("--log-speculation-detail-max-steps exceeds 2^20");
            }
            options.speculation_detail_max_steps = static_cast<std::uint32_t>(steps);
        } else if (arg == "--turn-anchors") {
            options.turn_anchors = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--turn-anchors"), "turn-anchors"));
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--log-speculation-detail") {
            options.log_speculation_detail = true;
        } else if (arg == "--log-speculation-detail-max-steps") {
            const std::uint64_t steps =
                parse_u64(require_value("--log-speculation-detail-max-steps"),
                          "log-speculation-detail-max-steps");
            if (steps == 0 || steps > kMaximumSpeculationDetailMaxSteps) {
                throw std::invalid_argument("--log-speculation-detail-max-steps must be in [1," +
                                            std::to_string(kMaximumSpeculationDetailMaxSteps) +
                                            "]");
            }
            options.speculation_detail_max_steps = static_cast<std::uint32_t>(steps);
            detail_steps_explicit                = true;
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device  = parse_nonnegative_int(require_value("--device"), "device");
            device_explicit = true;
        } else if (arg == "--tp") {
            options.tp = product::parse_tp(require_value("--tp"));
        } else if (arg == "--devices") {
            options.devices = product::parse_devices(require_value("--devices"));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--rope-yarn-factor") {
            options.yarn.factor =
                parse_float_in(require_value(arg.c_str()), arg.c_str(), 1.0F, 4.0F);
        } else if (arg == "--rope-original-max-position") {
            options.yarn.original_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value(arg.c_str()), arg.c_str()));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-device") {
            options.vision_device = product::parse_device_id(require_value("--vision-device"));
        } else if (arg == "--max-vision-tokens") {
            options.max_vision_tokens =
                product::parse_max_vision_tokens(require_value("--max-vision-tokens"));
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-tp-mailbox") {
            options.tp_mailbox = false;
        } else if (arg == "--embedding-host") {
            options.embedding_host = true;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (options.log_speculation_detail && options.request_log_jsonl.empty()) {
        throw std::invalid_argument("--log-speculation-detail requires --request-log-jsonl");
    }
    if (detail_steps_explicit && !options.log_speculation_detail) {
        throw std::invalid_argument(
            "--log-speculation-detail-max-steps requires --log-speculation-detail");
    }
    if (vram_headroom_mib.has_value()) {
        if (options.kv_capacity.mode != KvCapacityMode::Automatic) {
            throw std::invalid_argument("--vram-headroom-mib requires --kv-capacity auto");
        }
        options.kv_capacity = KvCapacityPolicy::automatic(*vram_headroom_mib << 20);
    }

    options.context_cache.enabled = options.allow_prefix_reuse;

    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    product::resolve_tensor_parallel_devices(options.tp, options.devices, options.device,
                                             device_explicit);
    product::validate_vision_options(options.enable_vision, options.vision_device,
                                     options.max_vision_tokens, options.devices);
    if (options.tp == 2) {
        // Rank 1's KV pages and StateImages have no Host tier: a tensor-parallel server keeps every
        // checkpoint in Device StateImages. This tree's single Host budget knob is the one that must
        // be 0; the Device StateImage and private-catalog floors come from the Engine defaults, which
        // keep a long prompt's endpoint and anchors resident together.
        if (host_capacity_explicit) {
            throw std::invalid_argument("--host-context-mib must be 0 with --tp 2");
        }
        options.context_cache.host_capacity_bytes = 0;
    }

    runtime::validate_yarn(options.yarn, options.max_context);
    product::validate_speculative_cli_options(options.speculative);
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    return std::string(artifact_model_name);
}

} // namespace ninfer::serve
