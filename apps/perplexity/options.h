#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::perplexity {

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::uint32_t context               = 4096;
    std::uint32_t stride                = 2048;
    int device                          = 0;
    int tp                              = 1;
    std::vector<int> devices; // One id per tensor-parallel rank; {device} at tp 1.
    ninfer::KvCacheStorage kv           = ninfer::KvCacheStorage::Fp8E4M3Row256;
    bool quick                          = false;
    bool embedding_host                 = false;
    ninfer::product::LogLevel log_level = ninfer::product::LogLevel::Info;
};

[[nodiscard]] std::string usage_text();

// Throws std::invalid_argument for a malformed command line.
[[nodiscard]] Options parse_options(int argc, char** argv);

} // namespace ninfer::perplexity
