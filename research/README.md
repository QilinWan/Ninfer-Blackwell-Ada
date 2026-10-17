# NInfer two-GPU (tp 2) research record — status 2026-10-18

This branch (`blackwell-only`) is the Blackwell-only snapshot of a private research fork of
[Neroued/ninfer](https://github.com/Neroued/ninfer). It carried two goals: an experimental
two-GPU tensor-parallel mode (`--tp 2`) so that Qwen3.8-27B NVFP4 fits two 16 GB boards, and an
Ada (sm_89) port. **The Ada port was removed from this branch** (2026-10-18) and lives in
`tp2/raw`; everything below is the sm_120a record.

The project is **on hold**: the text face is measured and reproducible, but the hardware target it
was built for cannot reach the planned operating point. The reasons are recorded in §3 so that a
future session does not re-derive them.

## 1. What works today (measured on 2x RTX 5060 Ti 16 GB, sm_120a, 36 SM, no P2P)

Artifact: `Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer` (23,773,675,264 B,
sha256 `3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808`), `--kv-dtype nvfp4`,
CUDA 13, commit `95884c20`:

| Item | Measured |
|---|---|
| Context | native **262,144** with MTP3 + vision (free 1.03 GiB) or without vision (2.01 GiB) |
| Concurrency | **C=8** at 262,144 (free 1.12 GiB); 8 lanes measured at **217.5 tok/s aggregate** |
| MTP3 speed | 52.4 tok/s, accept 72.18%, 59.6 ms/round (K=5: 57.0 tok/s) |
| DFlash2 K=7 speed | **140.4 tok/s**, accept 62.09%, 37.5 ms/round |
| GSM8K 500 (thinking off) | 482/500 = **96.4%** canonical (nvfp4 KV), 484/500 (int8) |
| Perplexity | 4.267108 (nvfp4 KV) vs 4.259926 (int8), 1580.7 tok/s both |
| Verbatim | greedy 60/60 and dflash2 10/10 identical across runs and KV types |
| Capacity | 196,608 @ int8 KV; 262,144 @ nvfp4 KV; DFlash2 max ~180,224 (C=1) |
| Media at tp 2 | rejected by design: HTTP 500 "multimodal prefill is not implemented at tensor-parallel width 2" |

## 2. Two defects this window found and fixed (`95884c20`)

They are worth remembering because the two-device suites skip on one board, so `ctest` never
reached either path:

1. **MTP Finish phase never ran.** Upstream's Forward/Finish split guards the verification with
   `if (phase == SpeculativePhase::Forward) { ...; return; }`; the tensor-parallel merge left the
   `return` outside that guard (`program/speculative/mtp.cpp`), so every phase returned after the
   target forward and acceptance/draft/egress never ran. Symptom: `MTP batch returned invalid row
   metadata` at warmup, zero licensed counts every round.
2. **DFlash2 verification card had no `TpExecution`.** `execution/draft.cpp` built the card without
   the pair, so the rank-pair verification threw "requires tensor parallelism" on a two-device
   engine.

Both fixed and verified by `ninfer_qwen3_5_engine_mtp_tp2_real_test` / `..._dflash2_tp2_real_test`
(MTP accept 152/276 = 55.07%, accepted-lengths 81 steps = 81 rounds, prefix reuse 8302/8331;
DFlash2 accept 8/12).

## 3. Why the project is on hold (the hardware wall)

The target was: NVFP4 weights + nvfp4 KV, **512K context, DFlash2-class speed, 4 concurrent
lanes**, on 2x16 GB. On these cards it is not reachable, for four independent reasons:

1. **512K needs YaRN** (the model is native 262,144). The engine supports factor 2.0, but quality
   at 2x is unvalidated on this artifact.
2. **DFlash2's drafter costs ~1.75 GiB on rank 0**, so its usable context is ~180K at C=1 and it
   cannot hold C=4 above ~32K. MTP is the only backend that reaches 262K with concurrency.
3. **Even at a 17 GB artifact** (the all-nvfp4 target), the arithmetic gives: MTP 512K + C=4 fits
   (~1.7 GiB free); **DFlash2 512K + C=4 does not** (~0.5 GiB short). 512K and DFlash2 speed are
   mutually exclusive on 16 GB boards.
4. **No P2P**: every round pays host-staged all-reduce, which is why MTP runs at 52 tok/s where
   the donor's 2x5070 Ti measured 76-86.

The honest conclusion: this work needs a **single >=32 GB Blackwell card at tp 1** (multimodal
native, no all-reduce, no split state) or a 96 GB part for 512K + DFlash2 + concurrency. The
two-16-GB route is a dead end for the intended operating point, which is why it is archived
rather than continued.

## 4. Removing Ada (what changed, what stayed)

Removed: `cmake/Sm89Compat.cmake`, `src/ops/sm89/`, the `NINFER_SM89_BUILD` architecture switch,
the `NINFER_SM89` guards in `core/pdl.cuh`, `ops/common/mma.cuh`, the bf16 GDN kernel,
`program/planning/startup.cpp`; `tests/CMakeLists.txt`'s sm89 gate, `kv_storage_supported`'s
sm89 branch; `Dockerfile.sm89`, `tools/verify_sm89.sh`, `scripts/check-sm89-exclusions.py`,
`docs/sm89.md`(zh), `docs/maintainer/rtx4080s-32g/`, `docs/maintainer/sm89-e8-verification/`,
the e8-yarn-audit handoffs, and the Ada release notes.

Kept: the `rk4v4-e8` KV codec sources are still in the tree but unreachable (the Blackwell build
drops the E8 routes in `cmake/Sm120Compat.cmake` and the capability gate refuses the KV type);
they can be deleted in a separate cleanup.

The Ada-era build was already red before removal: configure failed on two stale test names and
~15 FP8 A8 translation units used TMA schedules that need sm_90 (`.tile requires .target sm_90`),
while the pre-rebase branch compiled them with a non-TMA schedule. That regression is the concrete
reason the Ada port was judged not worth maintaining.

## 5. How to resume

Environment this was developed on: Ubuntu 22.04, CUDA 13.x, CMake >= 3.28 (4.x), Ninja, a private
deps prefix (`../deps-ada`) with libcurl >= 7.85 and FFmpeg >= 6 (vision ON), two sm_120 boards.

```bash
# deps prefix (one time): libcurl 8.11 + FFmpeg 6.1.6 built with --prefix=../deps-ada
cmake -S . -B build-120a-cu13 -G Ninja -DCMAKE_BUILD_TYPE=Release       -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_ENABLE_VISION=ON -DBUILD_TESTING=ON       -DNINFER_BUILD_BENCHMARKS=ON -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON -DCMAKE_PREFIX_PATH=../deps-ada
cmake --build build-120a-cu13 -j14
# the local convenience wrapper is scripts/build-ada.sh (historical name; 120a only)
```

Gate (the full pre-release gate; `relay/gate-steps.sh` in this archive):
`GATE_PATCHES=... bash research/scripts/gate-steps.sh <sha> <label>` builds a worktree, then runs
the family subset (`linear|fp8|nvfp4|bf16`), the cluster gate (`-E softmax_attention`) and the
12/2 qualification face (`ctest -R softmax_attention` + three `--kv-dtype` runs). Green numbers at
`d01dbcf7`: subset 28/28 425.90 s; cluster 174/174 1056.48 s (138 passed / 36 skipped / 0 failed);
qualification 1/1 757.47 s.

Serve (the measured production line):

```bash
./build-120a-cu13/apps/ninfer-serve <artifact>.ninfer --host 127.0.0.1 --port 8090 --tp 2 \
  --devices 0,1 --kv-dtype nvfp4 --max-context 262144 --kv-capacity 262144 \
  --device-state-slots 8 --max-concurrency 8 --spec mtp --draft-tokens 3 --lm-head-draft \
  --request-log-jsonl req.jsonl --log-speculation-detail
```

Perplexity: `apps/ninfer-perplexity <artifact> --corpus eval/corpora/perplexity-1m/manifest.json
--quick --tp 2 --devices 0,1 --kv-dtype nvfp4 --context 4096 --stride 2048 --output <empty dir>`.

## 6. Where the evidence is

- `replay-log-20261016.md` — the long ledger: per-commit replay decisions, every gate run, every
  trap (§82-84 are this window).
- `s7-seed-5060ti.md` — the tuning/capacity write-back: per-rank weights, graph allowances,
  context x concurrency matrix, KV-type cost, sweep, GSM8K.
- `RESULTS.md` — one-page summary of every measured number.
- `cards/` — the handoff prompts (v27, v28) as they were used.
- `scripts/` — the runnable harness: `gate-steps.sh`, `s3-bisect.sh`, `window-d-setup.sh`,
  `finish-everything.sh`, `check_accepted_lengths.py`, `d2_extract.py`, `gpu_rank_mem.sh`,
  `passset_from_log.py`, `gsm8k/{run_gsm8k.py,grade.py}`.
- `evidence/` — the logs the numbers come from (gate chain, qualification, two-device suites,
  record/compare, GSM8K reports, perplexity, sweep).

## 7. Open items / traps for the next session

- The `gate.sh` perplexity leg hard-codes `--kv-dtype int8`; do not smuggle `--kv-dtype nvfp4` in
  through `GATE_EXTRA_FLAGS` (it also hits the golden leg and changes the S3 baseline).
- `--spec mtp` accepts `--draft-tokens` 1..5 only; DFlash2 takes 7.
- `--kv-capacity` is one shared pool, not per lane: C=4 at 512K means four conversations share
  512K tokens.
- `pkill -f <pattern>` matches the shell command that contains the pattern; kill by PID.
- The two-device suites skip with code 77 on a single card, so a green one-card `ctest` says
  nothing about tp 2; the MTP/DFlash2 defects above lived exactly there.
- The NF4 artifact (`qwen3.8-27b-lynn-nf4.ninfer`) cannot load at tp 2: its output head is
  `q6_g64 [248320,5120]` and q6 has no half-vocabulary shape (`q6 linear: unsupported shape`).
  Its DFlash2 component was also damaged during conversion.
