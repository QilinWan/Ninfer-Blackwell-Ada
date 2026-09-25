// tp 1 golden runner: loads an artifact on one device through the public Engine API, decodes a
// deterministic byte-id prompt greedily and prints the generated token ids. The same source
// builds against this fork and against upstream ninfer (public headers only), so the two
// binaries' outputs can be compared token for token; see tools/golden/README.md.
#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::uint64_t splitmix(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " ARTIFACT --prompt-tokens N [--seed S] [--max-new N] [--max-context N]"
                 " [--kv-dtype bf16|int8] [--prefill-chunk N]\n";
    std::exit(2);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); }
    ninfer::EngineOptions options;
    options.artifact_path     = argv[1];
    std::uint32_t prompt_size = 0;
    std::uint64_t seed        = 7;
    std::uint32_t max_new     = 128;
    std::uint32_t max_context = 4096;
    options.prefill_chunk     = 1024;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value           = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); }
            return argv[++i];
        };
        if (arg == "--prompt-tokens") {
            prompt_size = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--seed") {
            seed = std::stoull(value());
        } else if (arg == "--max-new") {
            max_new = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--max-context") {
            max_context = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--kv-dtype") {
            const std::string dtype = value();
            if (dtype == "bf16") {
                options.kv_cache = ninfer::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
            } else {
                usage(argv[0]);
            }
        } else {
            usage(argv[0]);
        }
    }
    if (prompt_size == 0 || prompt_size + max_new > max_context) { usage(argv[0]); }
    options.max_context     = max_context;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency = 1;

    // Plain byte ids only (0..255): the synthetic tokenizer's specials start at 256.
    std::vector<ninfer::TokenId> prompt(prompt_size);
    std::uint64_t state = seed;
    for (auto& token : prompt) { token = static_cast<ninfer::TokenId>(splitmix(state) % 256U); }

    try {
        ninfer::Engine engine(std::move(options));
        ninfer::RequestOptions request;
        request.execution.sampling.temperature    = 0.0F; // exact argmax
        request.execution.requested_output_tokens = max_new;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = true;
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(prompt, false), request);
        std::cout << "prompt_tokens " << prompt.size() << '\n';
        std::cout << "generated_tokens " << result.generated_token_ids.size() << '\n';
        std::cout << "finish_reason " << static_cast<int>(result.finish_reason) << '\n';
        std::cout << "ids";
        for (const auto id : result.generated_token_ids) { std::cout << ' ' << id; }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tp1_golden: " << error.what() << '\n';
        return 1;
    }
}
