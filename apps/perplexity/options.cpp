#include "options.h"
#include "product/tensor_parallel_options.h"

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::perplexity {
namespace {

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

} // namespace

std::string usage_text() {
    return "usage: ninfer-perplexity <model.ninfer> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
           "       [--context N] [--stride N] [--device N] [--tp 1|2 --devices A,B]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--embedding-host]\n"
           "       [--output <directory>]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "--tp 2 --devices A,B scores on two GPUs (rank 0 on A); it needs --kv-dtype bf16|int8|nvfp4.\n"
           "--embedding-host reads the token embedding table in place from pinned host memory.\n";
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        return Options{.help_requested = true};
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact         = argv[1];
    bool device_explicit = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value("--context"), "context");
        } else if (option == "--stride") {
            out.stride = parse_integer<std::uint32_t>(value("--stride"), "stride");
        } else if (option == "--device") {
            out.device      = parse_integer<int>(value("--device"), "device");
            device_explicit = true;
        } else if (option == "--tp") {
            out.tp = ninfer::product::parse_tp(value("--tp"));
        } else if (option == "--devices") {
            out.devices = ninfer::product::parse_devices(value("--devices"));
        } else if (option == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") {
                out.kv = ninfer::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                out.kv = ninfer::KvCacheStorage::Int8Group64;
            } else if (dtype == "fp8") {
                out.kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
            } else if (dtype == "nvfp4") {
                out.kv = ninfer::KvCacheStorage::Nvfp4Group16;
            } else if (dtype == "k8v4") {
                out.kv = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
            } else {
                usage_error("--kv-dtype must be bf16, int8, fp8, nvfp4, or k8v4");
            }
        } else if (option == "--embedding-host") {
            out.embedding_host = true;
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--log-level") {
            out.log_level = ninfer::product::parse_log_level(value("--log-level"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    ninfer::product::resolve_tensor_parallel_devices(out.tp, out.devices, out.device,
                                                     device_explicit);
    return out;
}

} // namespace ninfer::perplexity
