// Class-body fragment of ProgramImpl (program/program_impl.h), included at the end of its private
// section: the two-device (tp 2) helpers. Definitions in program/tp2/program_impl_tp2.inc and
// the other program/tp2 fragments. Not a header; our file, see docs/maintainer/upstream-merge.md.

private:
    // Constructor steps at tensor-parallel width 2: the width checks and rank 1's runtime, events
    // and mailbox (before rank 0's storage is carved), then rank 1's TpExecution binding (after
    // rank 0's round state exists).
    void construct_tensor_parallel(const SequencePlanImpl& plan, ExecutionContext* execution_in,
                                   const execution::Parameters* peer_parameters_in);
    void bind_tensor_parallel_execution();
    // The Device slot of the StateImage holding `sequence`'s committed continuation (the Fork
    // source while a Fork is pending); empty when either bound image is Host-only.
    [[nodiscard]] std::optional<std::int32_t>
    committed_state_slot(const SequenceState& sequence) const;
    // Rank 1's copy of an I32 control scalar, uploaded on rank 1's stream.
    void set_peer_i32(Tensor& tensor, std::int32_t value);
    void attach_tensor_parallel_mirrors();
    // Tensor-parallel causal scoring: the complete [V,C] logits of rank 0's staged hidden
    // columns [H,C] through the vocabulary-split output head, gathered on rank 0.
    void project_score_tile_split(const Tensor& hidden, const Tensor& logits);

    // True when rank 1 keeps its own copy of every retained target hidden: at tensor-parallel
    // width 2 under MTP, whose bridge reads it on both ranks.
    [[nodiscard]] bool peer_retains_hidden() const noexcept {
        return peer != nullptr && speculative_backend == SpeculativeBackend::Mtp;
    }

    // Rank 1's copy of `sequence.tail_hidden` (peer_retains_hidden() only).
    [[nodiscard]] Tensor peer_tail_hidden(const SequenceState& sequence) const;
    // Uploads the active sequence's Text and backend KV execution rows into the prefill control
    // scalars of every rank.
    void publish_kv_rows(const SequenceState& sequence);
