#pragma once

// Request-log measurement of how much of a response repeats its own prompt: the share of a
// copy-style draft (prompt lookup) could propose. Computed by Serve after generation, only with
// --log-speculation-detail.

#include "ninfer/types.h"

#include <cstddef>
#include <optional>
#include <span>

namespace ninfer::serve {

// Token n-gram order of prompt_ngram_overlap.
inline constexpr std::size_t kPromptOverlapNgram = 4;

// Fraction of the output's overlapping 4-token windows (output.size() - 3 of them, repeats counted
// once per position) whose exact token sequence also occurs somewhere in the prompt. Every output
// token counts, thinking and tool-call tokens included. nullopt when the output has fewer than four
// tokens; 0 when the prompt does.
[[nodiscard]] std::optional<double> prompt_ngram_overlap(std::span<const TokenId> prompt,
                                                         std::span<const TokenId> output);

} // namespace ninfer::serve
