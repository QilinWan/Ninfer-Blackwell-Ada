#include "options.h"
#include "product/tensor_parallel_options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures      = 0;
    const auto choice = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--choice", "",
                               "--choice", "a|b", "--choice", "你好"});
    failures += check(choice.choices == std::vector<std::string>{"", "a|b", "你好"},
                      "choice flags changed literal text or order");
    failures +=
        check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", ""}).regex == "",
              "empty regex must be distinct from no regex");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", "",
                                       "--choice", "yes"});
                      }),
                      "regex and choice were accepted together");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex",
                                       "a", "--json-object"});
                      }),
                      "regex and JSON were accepted together");
    failures += check(
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--grammar-file", "answer.gbnf"})
                .grammar_path == "answer.gbnf",
        "grammar file argument was lost");
    failures +=
        check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json-object"}).json_object,
              "JSON object CLI option lost");
    failures += check(
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json-schema-file", "answer.json"})
                .json_schema_path == "answer.json",
        "schema file CLI option lost");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello",
                                       "--json-object", "--grammar-file", "a.gbnf"});
                      }),
                      "conflicting CLI constraints accepted");
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(help.find("--vram-headroom-mib") != std::string::npos,
                      "CLI help omits the VRAM headroom control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    const ninfer::cli::Options single = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"});
    failures += check(single.tp == 1 && single.devices == std::vector<int>{0},
                      "CLI default is not one rank on device 0");
    const ninfer::cli::Options split =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--tp", "2", "--devices", "1,0"});
    failures += check(split.tp == 2 && split.devices == std::vector<int>{1, 0} && split.device == 1,
                      "--tp 2 --devices did not select both ranks with rank 0 first");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--tp", "2"});
              }),
              "--tp 2 was accepted without --devices");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--tp",
                                       "2", "--devices", "0"});
                      }),
                      "--devices with fewer ids than --tp was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--device", "1", "--tp", "2", "--devices", "0,1"});
                      }),
                      "--device disagreeing with the rank 0 device was accepted");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--tp", "3"});
              }),
              "--tp 3 was accepted");
    failures += check(help.find("--tp") != std::string::npos, "CLI help omits --tp");
    failures +=
        check(split.tp_mailbox && !parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--tp",
                                          "2", "--devices", "1,0", "--no-tp-mailbox"})
                                       .tp_mailbox,
              "--no-tp-mailbox did not disable the captured mailbox transport");
    const ninfer::cli::Options vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--tp", "2", "--devices", "1,0",
               "--vision", "--vision-device", "0", "--max-vision-tokens", "1024"});
    failures += check(vision.vision_device == 0 && vision.max_vision_tokens == 1024U &&
                          !split.vision_device && !split.max_vision_tokens,
                      "--vision-device and --max-vision-tokens were not parsed");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--tp", "2",
                               "--devices", "0,1", "--vision", "--vision-device", "3"});
              }),
              "--vision-device outside --devices was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--max-vision-tokens", "1024"});
                      }),
                      "--max-vision-tokens was accepted without --vision");

    // S6 capability gate: the pure verdicts behind a `--tp 2` startup rejection, asserted without a
    // CUDA device. CC 12.0 passes; every other capability, Ada's 8.9 (sm_89) included, answers with
    // a reason naming the requirement, the observed capability and the single-device route. The
    // Engine calls these same functions once it owns the two rank DeviceContexts, so the CPU verdict
    // and the startup message cannot drift apart.
    failures += check(!ninfer::product::tensor_parallel_capability_verdict(12, 0),
                      "CC 12.0 was not accepted for --tp 2");
    const auto ada = ninfer::product::tensor_parallel_capability_verdict(8, 9);
    failures += check(ada.has_value() && ada->find("sm_120") != std::string::npos &&
                          ada->find("8.9") != std::string::npos &&
                          ada->find("--tp 1") != std::string::npos,
                      "Ada (CC 8.9) --tp 2 rejection is not a readable reason");
    failures += check(ninfer::product::tensor_parallel_capability_verdict(12, 1).has_value(),
                      "a CC other than 12.0 was accepted for --tp 2");

    // The three KV storages with a two-device route pass: bf16, int8-G64 and nvfp4-G16 (the
    // head-local 12/2 geometry is instantiated for those kernels, softmax_attention.h).
    failures += check(!ninfer::product::tensor_parallel_kv_verdict(ninfer::KvCacheStorage::BFloat16) &&
                          !ninfer::product::tensor_parallel_kv_verdict(
                              ninfer::KvCacheStorage::Int8Group64) &&
                          !ninfer::product::tensor_parallel_kv_verdict(
                              ninfer::KvCacheStorage::Nvfp4Group16),
                      "a KV storage with a tensor-parallel route was rejected for --tp 2");
    const auto e8 = ninfer::product::tensor_parallel_kv_verdict(ninfer::KvCacheStorage::RK4V4E8);
    failures += check(e8.has_value() && e8->find("rk4v4-e8") != std::string::npos &&
                          e8->find("--tp 1") != std::string::npos,
                      "rk4v4-e8 KV passed the --tp 2 gate without naming the storage");
    for (const ninfer::KvCacheStorage storage : {ninfer::KvCacheStorage::Fp8E4M3Row256,
                                                 ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                                                 ninfer::KvCacheStorage::RK4V4E8}) {
        failures += check(ninfer::product::tensor_parallel_kv_verdict(storage).has_value(),
                          "an FP8 or 4-bit KV storage passed the --tp 2 gate");
    }
    return failures == 0 ? 0 : 1;
}
