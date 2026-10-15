// Class-body fragment of TextContext (execution/text.h), included at the end of its private
// section: the two-device (tp == 2) schedule and its state. The definitions are in
// tp2/text_tp2.cpp. Not a header; our file, see docs/maintainer/upstream-merge.md.

private:
    // --- tp == 2 -------------------------------------------------------------------------------
    // Separate functions rather than branches inside the single-device ones, so the tp1 schedule
    // is unchanged. The residual x[r] is replicated: each layer ends in an all-reduce that leaves
    // the identical BF16 sum on both ranks, so every later per-rank stage sees identical inputs.
    // Rank r's work is issued on ExecutionContext::dev[r]'s stream; only the collectives inside
    // the row-parallel projections order the two streams, with no host synchronization.
    using RankBlocks = std::array<const BlockParameters*, kTensorParallelWidth>;

    [[nodiscard]] bool tp2() const noexcept { return tp_ != nullptr; }

    void validate_tensor_parallel();
    void require_single_device(const char* operation) const;
    [[nodiscard]] const Parameters& rank_parameters(int rank) const noexcept;
    [[nodiscard]] std::array<WorkspaceArena*, kTensorParallelWidth> workspaces() const noexcept;
    [[nodiscard]] cudaStream_t rank_stream(int rank) const noexcept;
    // Stream and SM count of `rank`'s device, for Ops whose launch shape depends on the device.
    [[nodiscard]] DeviceExecutionView rank_execution_view(int rank) const noexcept;
    [[nodiscard]] LinearAttentionStatePool& rank_state(int rank) const noexcept;
    [[nodiscard]] const qwen3_5::PagedKVCache& rank_text_cache(int rank) const;
    // Rank 0 reads the active_* bindings exactly as the single-device path does; rank 1 reads
    // the peer_* bindings the tp2 entry point sets from rank 1's own control tensors.
    [[nodiscard]] const Tensor& rank_cache_positions(int rank) const;
    [[nodiscard]] const Tensor& rank_rope_positions(int rank) const;
    [[nodiscard]] const Tensor& rank_kv_table_rows(int rank) const;
    [[nodiscard]] const Tensor& rank_linear_state_source_slots(int rank) const;
    [[nodiscard]] const Tensor& rank_linear_state_destination_slots(int rank) const;
    // Both ranks bind valid columns, or neither does (an empty Tensor: every column is live); two
    // ranks masking different columns would diverge silently.
    [[nodiscard]] Tensor rank_valid_columns(int rank) const;
    // The MTP execution rows: the batch binding of an MTP decode call, else the prefill scalar.
    [[nodiscard]] const Tensor& rank_backend_kv_table_rows(int rank) const;
    [[nodiscard]] const qwen3_5::PagedKVCache& rank_mtp_cache(int rank) const;
    [[nodiscard]] qwen3_5::PagedKVCacheView rank_mtp_kv(int rank) const;
    [[nodiscard]] const MtpParameters& rank_mtp(int rank) const;
    [[nodiscard]] const GdnReplayRecords& rank_replay_records(int rank) const;
    void validate_tensor_parallel_mtp();
    void attn_mix_tp2(const RankBlocks& weights, RankTensors& x, int index,
                      const RankTensors& staging);
    void gdn_mix_tp2(const RankBlocks& weights, RankTensors& x, int index, Phase phase,
                     const RankTensors& staging);
    void mlp_tail_tp2(const RankBlocks& weights, RankTensors& x, const RankTensors& staging);
    template <class Tap>
    void run_layers_tp2(RankTensors& x, Phase phase, const RankTensors& staging, Tap& tap);
    template <class Tap>
    void target_verify_batch_tp2_impl(const RankTensors& ids, const RankTensors& cache_positions,
                                      const RankTensors& rope_positions,
                                      const RankTensors& valid_columns,
                                      const RankTensors& kv_table_rows,
                                      const RankTensors& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      const RankTensors& hidden, Tensor& logits,
                                      Tensor& target_tokens, Tap& tap);
    // Vocabulary-split head: each rank projects its half of the vocabulary from its final hidden
    // columns, and rank 0 alone gathers the complete [V, C] logits into its `logits` (one pull of
    // rank 1's half).
    void logits_tp2(const RankTensors& hidden, Tensor& logits);
    void logits_tp2(const RankTensors& hidden, const std::array<const LinearParameters*, 2>& head,
                    Tensor& logits);
    // MTP head over both ranks: three all-reduces (input projection, attention output, post-mixer
    // down projection) leave the replicated residual identical on the two ranks, as in the text
    // layers. The attention runs over each rank's half of the heads and its own MTP KV pages.
    // `input_embeddings`, when set, is rank 0's composed embedding of `ids`.
    void mtp_forward_stem_tp2(const Tensor& ids, const RankTensors& hidden,
                              const Tensor* input_embeddings, RankTensors& x, RankTensors& ah,
                              const RankTensors& staging);
    void mtp_forward_tail_tp2(RankTensors& x, const RankTensors& ah, const RankTensors& positions,
                              const RankTensors& rope_positions,
                              ops::CausalAttentionExecutionEnvelope envelope,
                              const RankTensors& mtp_hidden, const RankTensors& staging);
    void mtp_forward_core_tp2(const Tensor& ids, const RankTensors& hidden,
                              const RankTensors& positions, const RankTensors& rope_positions,
                              ops::CausalAttentionExecutionEnvelope envelope,
                              const RankTensors& mtp_hidden, const Tensor* input_embeddings);
    void mtp_prefill_chunk_tp2(const Tensor& ids, const RankTensors& hidden,
                               const Tensor* input_embeddings, const RankTensors& positions,
                               const RankTensors& rope_positions,
                               ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                               const RankTensors* final_hidden, Tensor* logits,
                               Tensor* draft_token);
    // The optimized proposal head proposes on rank 0 alone when PrimaryOnly, or on both ranks when
    // split by vocabulary (execution::proposal_argmax_split, rank 0 combining the two argmaxes);
    // the full head is the vocabulary-split output head, gathered before rank 0's argmax.
    void proposal_argmax_tp2(const RankTensors& hidden, Tensor& logits, Tensor& proposal_tokens);
    void ordinary_decode_batch_tp2(const Tensor& ids, const Tensor& cache_positions,
                                   const Tensor& rope_positions, const Tensor& kv_table_rows,
                                   const Tensor& linear_state_source_slots,
                                   const Tensor& linear_state_destination_slots,
                                   ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                   Tensor& logits);

    // tp2 prefill of one text or multimodal chunk (exactly one of `text_prefill` and
    // `multimodal`). A multimodal chunk scatters the Vision session's merged embeddings into both
    // ranks' residual, each from its own copy. A DFlash feature tap reads rank 0's replicated
    // residual.
    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl_tp2(std::span<const int> ids, const TextPrefill* text_prefill,
                     const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);

    // rank 1's StateImage column for set_rewrite_checkpoint_hidden_output() (MTP only).
    Tensor* peer_rewrite_checkpoint_hidden_output_ = nullptr;
    // The split DFlash2 drafter's rank 1 feature sink (set_peer_feature_sink); null otherwise.
    DFlashFeatureSink* peer_feature_sink_ = nullptr;
    // Rank 1's calls on peer_feature_sink_, beside the rank 0 tap's (no-ops without a peer sink).
    void peer_tap_begin(const Tensor& x1);
    void peer_tap_capture_layer(int layer, const Tensor& x1);
    void peer_tap_capture_positions(const Tensor& positions1);
    void peer_tap_consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);

    const TpExecution* tp_ = nullptr;
    // Each rank's share of config_ (shard_text_config), present only at tp == 2.
    std::optional<TextConfig> shard_config_;
    const Tensor* peer_cache_positions_                = nullptr;
    const Tensor* peer_rope_positions_                 = nullptr;
    const Tensor* peer_kv_table_rows_                  = nullptr;
    const Tensor* peer_linear_state_source_slots_      = nullptr;
    const Tensor* peer_linear_state_destination_slots_ = nullptr;
    const Tensor* peer_valid_columns_                  = nullptr;
    const Tensor* peer_backend_kv_table_rows_          = nullptr;
