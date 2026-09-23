#include "models/qwen3_5/execution/tp.h"

#include "models/qwen3_5/execution/linear.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::uint32_t divide(std::uint32_t extent, int width, const char* label) {
    const auto divisor = static_cast<std::uint32_t>(width);
    if (extent == 0 || extent % divisor != 0) {
        throw std::invalid_argument(std::string("tensor-parallel Text config: ") + label +
                                    " is not divisible by the tensor-parallel width");
    }
    return extent / divisor;
}

} // namespace

TextConfig shard_text_config(const TextConfig& config, int width) {
    if (width != 1 && width != kTensorParallelWidth) {
        throw std::invalid_argument("tensor-parallel Text config: width must be 1 or 2");
    }
    TextConfig out = config;
    if (width == 1) { return out; }
    const auto* dense = std::get_if<DenseConfig>(&out.ffn);
    if (dense == nullptr) {
        throw std::invalid_argument("tensor-parallel Text config: the MoE FFN has no two-device "
                                    "placement");
    }
    out.ffn        = DenseConfig{divide(dense->intermediate_size, width, "intermediate_size")};
    out.vocab_size = divide(out.vocab_size, width, "vocab_size");
    if (out.attention) {
        out.attention->num_attention_heads =
            divide(out.attention->num_attention_heads, width, "num_attention_heads");
        out.attention->num_key_value_heads =
            divide(out.attention->num_key_value_heads, width, "num_key_value_heads");
    }
    if (out.gdn) {
        out.gdn->linear_num_key_heads =
            divide(out.gdn->linear_num_key_heads, width, "linear_num_key_heads");
        out.gdn->linear_num_value_heads =
            divide(out.gdn->linear_num_value_heads, width, "linear_num_value_heads");
    }
    return out;
}

std::size_t output_head_split_workspace_bytes(const LinearParameters& shard, std::int32_t first,
                                              std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(shard.weight.qtype, shard.weight.n, shard.weight.k,
                                                shard.policy, first, last);
}

void output_logits_split(const std::array<Tensor, 2>& hidden,
                         const std::array<const LinearParameters*, 2>& head,
                         const std::array<Tensor, 2>& partial, const std::array<Tensor, 2>& logits,
                         const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& execution, const ops::PeerEvents& events) {
    const std::int32_t columns = hidden[0].ne[1];
    const std::int32_t rows0   = partial[0].ne[0];
    const std::int32_t rows1   = partial[1].ne[0];
    for (std::size_t r = 0; r < 2; ++r) {
        if (partial[r].dtype != DType::BF16 || partial[r].ne[1] != columns ||
            !partial[r].is_contiguous() || logits[r].dtype != DType::BF16 ||
            logits[r].ne[0] != rows0 + rows1 || logits[r].ne[1] != columns ||
            logits[r].ne[2] != 1 || logits[r].ne[3] != 1 || !logits[r].is_contiguous()) {
            throw std::invalid_argument(
                "tensor-parallel logits: partial or gathered logits do not match the vocabulary");
        }
    }
    project_column_parallel(hidden, head, partial, workspace, execution);
    // The gather runs along ne[1] and the vocabulary is ne[0], so it runs one column at a time:
    // one column of a contiguous [V,C] BF16 matrix is a contiguous V-element run, which viewed as
    // [1,V] is the Op's [row length 1, row count V] layout. C is 1 in prefill and the decode batch
    // size otherwise.
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::array<Tensor, 2> piece{partial[0].slice(1, column, 1).view({1, rows0}),
                                          partial[1].slice(1, column, 1).view({1, rows1})};
        const std::array<Tensor, 2> whole{logits[0].slice(1, column, 1).view({1, rows0 + rows1}),
                                          logits[1].slice(1, column, 1).view({1, rows0 + rows1})};
        ops::allgather_rows(whole, piece, execution, events);
    }
}

OrdinaryPeerFrame ordinary_peer_frame(const qwen3_5::OrdinaryDecodeState& frame) {
    return {frame.tokens,
            frame.cache_positions,
            frame.rope_positions,
            frame.text_kv_table_rows,
            frame.state_source_slots,
            frame.state_destination_slots};
}

} // namespace ninfer::models::qwen3_5::execution
