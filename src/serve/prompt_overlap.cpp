#include "serve/prompt_overlap.h"

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace ninfer::serve {
namespace {

using Gram = std::array<TokenId, kPromptOverlapNgram>;

Gram gram_at(std::span<const TokenId> tokens, std::size_t start) noexcept {
    return Gram{tokens[start], tokens[start + 1], tokens[start + 2], tokens[start + 3]};
}

std::uint64_t gram_hash(const Gram& gram) noexcept {
    const std::uint64_t low = static_cast<std::uint32_t>(gram[0]) |
                              static_cast<std::uint64_t>(static_cast<std::uint32_t>(gram[1])) << 32;
    const std::uint64_t high = static_cast<std::uint32_t>(gram[2]) |
                               static_cast<std::uint64_t>(static_cast<std::uint32_t>(gram[3]))
                                   << 32;
    std::uint64_t hash = low * 0x9E3779B97F4A7C15ULL ^ std::rotl(high * 0xC2B2AE3D27D4EB4FULL, 31);
    hash ^= hash >> 29;
    hash *= 0xBF58476D1CE4E5B9ULL;
    return hash ^ (hash >> 32);
}

enum class Slot : std::uint8_t {
    Empty,
    Output,
    Found,
};

// Open-addressing set of the distinct output n-grams at load <= 1/4, so a prompt window that is
// not in the output usually stops at its first, empty slot without touching the keys.
class OutputGrams {
public:
    explicit OutputGrams(std::size_t windows)
        : mask_(std::bit_ceil(4 * windows) - 1), keys_(mask_ + 1), slots_(mask_ + 1, Slot::Empty) {}

    std::size_t insert(const Gram& gram) {
        std::size_t index = gram_hash(gram) & mask_;
        while (slots_[index] != Slot::Empty) {
            if (keys_[index] == gram) { return index; }
            index = (index + 1) & mask_;
        }
        slots_[index] = Slot::Output;
        keys_[index]  = gram;
        ++distinct_;
        return index;
    }

    // Marks the n-gram found; returns false once every distinct output n-gram is found.
    bool mark(const Gram& gram) noexcept {
        std::size_t index = gram_hash(gram) & mask_;
        while (slots_[index] != Slot::Empty) {
            if (keys_[index] == gram) {
                if (slots_[index] == Slot::Output) {
                    slots_[index] = Slot::Found;
                    ++found_;
                }
                return found_ != distinct_;
            }
            index = (index + 1) & mask_;
        }
        return true;
    }

    [[nodiscard]] bool found(std::size_t index) const noexcept {
        return slots_[index] == Slot::Found;
    }

private:
    std::size_t mask_;
    std::vector<Gram> keys_;
    std::vector<Slot> slots_;
    std::size_t distinct_ = 0;
    std::size_t found_    = 0;
};

} // namespace

std::optional<double> prompt_ngram_overlap(std::span<const TokenId> prompt,
                                           std::span<const TokenId> output) {
    if (output.size() < kPromptOverlapNgram) { return std::nullopt; }
    const std::size_t windows = output.size() - kPromptOverlapNgram + 1;
    OutputGrams grams(windows);
    std::vector<std::uint32_t> slot_of(windows);
    for (std::size_t i = 0; i < windows; ++i) {
        slot_of[i] = static_cast<std::uint32_t>(grams.insert(gram_at(output, i)));
    }
    if (prompt.size() >= kPromptOverlapNgram) {
        const std::size_t prompt_windows = prompt.size() - kPromptOverlapNgram + 1;
        for (std::size_t j = 0; j < prompt_windows; ++j) {
            if (!grams.mark(gram_at(prompt, j))) { break; }
        }
    }
    std::size_t matched = 0;
    for (const std::uint32_t slot : slot_of) { matched += grams.found(slot) ? 1U : 0U; }
    return static_cast<double>(matched) / static_cast<double>(windows);
}

} // namespace ninfer::serve
