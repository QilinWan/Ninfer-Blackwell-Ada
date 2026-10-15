// Class-body fragment of ProgramImpl (program/program_impl.h), included in its public section
// right after `workspace_plan`: the two-device (tp 2) runtime. The position matters: rank 1's
// storage and the mailbox are declared before rank 0's storage and the graph families so that
// they are destroyed after them. Not a header; our file, see docs/maintainer/upstream-merge.md.

public:
    // Tensor-parallel rank 1 (width 2 only). It holds rank 0's workspace layout (with Vision, only
    // the tower's rank holds the encode region; WorkspacePlan::rank_capacity) and rank 0's
    // persistent layout without the masked drafter's state (SequencePlanImpl::peer_persistent) on
    // ExecutionContext::dev[1]; its KV page pools, KV execution tables and StateImages are mirrors
    // that rank 0's pools drive, so rank 0's bookkeeping names both ranks' storage. Rank 1 reads
    // the Text and MTP KV, the GDN/hidden StateImages, its ReplaySSM records, the prefill hidden
    // and KV rows, and the ordinary, MTP or DFlash2 decode frame. Declared before rank 0's
    // storage so it is destroyed after it: rank 0's execution tables hold rank 1's row leases
    // until they are destroyed, and rank 0's graphs hold rank 1 nodes and events.
    struct PeerRuntime {
        PeerRuntime(DeviceContext& device, const SequencePlanImpl& plan);

        [[nodiscard]] static const PersistentLayout& layout(const SequencePlanImpl& plan);

        DeviceContext& device;
        DeviceArena persistent;
        DeviceArena workspace_storage;
        WorkspaceArena work;
        std::unique_ptr<qwen3_5::DecoderState> decoder;
        std::unique_ptr<qwen3_5::StateImageDevicePool> state_images;
        std::optional<GdnReplayRecords> replay_records;
        std::optional<ops::GdnReplayFoldPlan> replay_fold;
        qwen3_5::RoundState io;
        Tensor prefill_hidden;
        // Causal scoring only: rank 1's copy of the staged hidden columns, the input of its half
        // of the vocabulary-split output head.
        Tensor score_hidden;
        // Views rank 1's ordinary decode frame; empty under a speculative backend.
        execution::OrdinaryPeerFrame ordinary;
        // The split DFlash2 drafter (load/sharding.h): rank 1's drafter state (its DFlash rings
        // live in state_images, mirrored with rank 0's) and its drafter operands, bound by
        // bind_tensor_parallel_execution(); empty otherwise.
        std::optional<DFlashPersistentState> dflash;
        std::optional<execution::DFlashPeerDrafter> drafter;
    };

    ExecutionContext* execution_context          = nullptr;
    const execution::Parameters* peer_parameters = nullptr;
    std::unique_ptr<PeerRuntime> peer;
    // The captured all-reduces' pinned-host transport (tp 2 with CUDA Graphs, unless disabled),
    // attached to peer_events. Declared before peer_events, which points to it, and before the
    // graph families, whose kernels address its slab: both are destroyed first.
    std::optional<ops::PeerMailbox> peer_mailbox;
    std::optional<ops::PeerEvents> peer_events;
    std::optional<DecodeGraphPeerBridge> graph_bridge;
    std::optional<execution::TpExecution> tp_execution;
    // Filled at construction: which transport the captured all-reduces ended up with.
    TpTransportStatus tp_transport_status;

    // Exchanges one probe payload through peer_mailbox inside a two-device graph before any
    // decode graph is captured. A hang (WSL2's GPU virtualization is the known case) or a slow
    // round trip drops the mailbox: peer_events falls back to the staged copies for good.
    // NINFER_TP_MAILBOX_PROBE=off skips the probe, =fail runs it with a missing peer (test aid).
    void probe_peer_mailbox();

    // Called by prepare_graphs() after a mailbox exchange timed out in a graph's first launch,
    // with both devices idle: discards every captured graph and steps the transport down once,
    // from the mailbox everywhere to the mailbox with the MTP draft phase on the staged copies
    // (MTP only), then to the staged copies everywhere. Returns false when nothing is left to
    // step down to. The replacement mailbox is fresh: the hung one's words have diverged.
    bool degrade_peer_mailbox();

    [[nodiscard]] bool tensor_parallel() const noexcept { return peer != nullptr; }

    // The Parameters that hold the Vision tower: rank 1's when it holds the tower, else rank 0's.
    [[nodiscard]] const execution::Parameters& vision_parameters() const noexcept {
        return vision_rank == 1 && peer_parameters != nullptr ? *peer_parameters : parameters;
    }

    [[nodiscard]] const execution::TpExecution* tp_binding() const noexcept {
        return tp_execution ? &*tp_execution : nullptr;
    }

    [[nodiscard]] const DecodeGraphPeerBridge* graph_peer_bridge() const noexcept {
        return graph_bridge ? &*graph_bridge : nullptr;
    }

    // A prefill call's copy of tp_execution naming `sequence`'s rank-1 MTP KV row; empty at
    // width 1.
    [[nodiscard]] std::optional<execution::TpExecution>
    prefill_tp_binding(const SequenceState& sequence) const;

    // Waits for rank 1 and then rank 0; at width 1 only rank 0. Throws std::runtime_error when a
    // captured mailbox exchange of the retired work timed out: the two ranks' results diverged.
    void synchronize_devices() const;

    // MemorySummary::cuda_graph_observed_bytes, measured once by prepare_graphs().
    std::array<std::size_t, 2> graph_observed_bytes{};
