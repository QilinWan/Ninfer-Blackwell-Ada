#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::perplexity::Options parse(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), {"ninfer-perplexity", "model.ninfer", "--text", "a.txt"});
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::perplexity::parse_options(static_cast<int>(argv.size()), argv.data());
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
    int failures = 0;
    const ninfer::perplexity::Options single = parse({});
    failures += check(single.tp == 1 && single.device == 0 && single.devices == std::vector<int>{0},
                      "default is not one rank on device 0");
    const ninfer::perplexity::Options one = parse({"--device", "1"});
    failures += check(one.tp == 1 && one.device == 1 && one.devices == std::vector<int>{1},
                      "--device did not select the single rank");
    const ninfer::perplexity::Options split =
        parse({"--tp", "2", "--devices", "1,0", "--kv-dtype", "int8"});
    failures += check(split.tp == 2 && split.devices == std::vector<int>{1, 0} &&
                          split.device == 1 && split.kv == ninfer::KvCacheStorage::Int8Group64,
                      "--tp 2 --devices did not select both ranks with rank 0 first");
    failures += check(parse({"--device", "0", "--tp", "2", "--devices", "0,1"}).device == 0,
                      "--device equal to the rank 0 device was rejected");
    failures += check(rejects([] { (void)parse({"--tp", "2"}); }),
                      "--tp 2 was accepted without --devices");
    failures += check(rejects([] { (void)parse({"--tp", "2", "--devices", "0"}); }),
                      "--devices with fewer ids than --tp was accepted");
    failures += check(rejects([] { (void)parse({"--devices", "0,1"}); }),
                      "--devices with two ids was accepted at tp 1");
    failures +=
        check(rejects([] { (void)parse({"--device", "1", "--tp", "2", "--devices", "0,1"}); }),
              "--device disagreeing with the rank 0 device was accepted");
    failures += check(rejects([] { (void)parse({"--tp", "3", "--devices", "0,1,2"}); }),
                      "--tp 3 was accepted");
    failures += check(rejects([] { (void)parse({"--tp"}); }), "--tp without a value was accepted");
    failures += check(ninfer::perplexity::usage_text().find("--tp 1|2 --devices A,B") !=
                          std::string::npos,
                      "help omits --tp and --devices");
    return failures == 0 ? 0 : 1;
}
