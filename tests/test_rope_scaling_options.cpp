#include "apps/cli/options.h"
#include "serve/serve_options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    int failures = 0;
    for (bool server : {false, true}) {
        const auto parse = [&](std::vector<std::string> flags) {
            std::vector<std::string> args{server ? "ninfer-serve" : "ninfer", "model.ninfer"};
            if (!server) { args.insert(args.end(), {"--prompt", "hello"}); }
            args.insert(args.end(), flags.begin(), flags.end());
            std::vector<char*> argv;
            for (auto& arg : args) { argv.push_back(arg.data()); }
            return server ? ninfer::serve::parse_serve_options(argv.size(), argv.data()).yarn
                          : ninfer::cli::parse_options(argv.size(), argv.data()).yarn;
        };
        const auto native = parse({});
        failures += native.factor != 1.0F || native.original_context != 262144;
        const auto scaled =
            parse({"--rope-yarn-factor", "1.5", "--rope-original-max-position", "262144",
                   "--max-context", "393216", "--vision", "--spec", "mtp", "--draft-tokens", "3"});
        failures += scaled.factor != 1.5F || scaled.original_context != 262144;
        for (const auto& flags : std::vector<std::vector<std::string>>{
                 {"--rope-yarn-factor", "nan"},
                 {"--rope-yarn-factor", "1.5junk"},
                 {"--rope-yarn-factor", "0.9"},
                 {"--rope-yarn-factor", "4.1"},
                 {"--rope-original-max-position", "1048576"},
                 {"--rope-yarn-factor", "1.5", "--max-context", "393217"}}) {
            bool rejected = false;
            try {
                (void)parse(flags);
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += !rejected;
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " CLI/server YaRN options\n";
    return failures ? 1 : 0;
}
