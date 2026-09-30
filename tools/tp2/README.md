# tools/tp2

- `gate.sh` — the behavior-preservation gate of the two-GPU layer: one command, PASS/FAIL per
  stage against a recorded reference. Mandatory for every upstream merge and every change to the
  tp2 layer ([Merging upstream](../../docs/maintainer/upstream-merge.md#4-the-gate-mandatory-for-every-merge-and-every-tp2-change)).
- `gate_client.py` — its serve-stage client (requests, request-log statistics, comparison).
- `check_shard_sources.py` — the static guard of the shards compiled from upstream's sources.
- `surface.sh` — the fork's footprint inside upstream's files: modified files, lines, hunks, the
  files with the most hunks (`tools/tp2/surface.sh [<upstream-ref> [<ref>]]`).
- `mailbox_probe.cu` — the standalone pinned-host mailbox check ([Tools](../README.md#standalone-tp2-mailbox-probe)).

## gate.sh

```bash
tools/tp2/gate.sh [--record] [--stages LIST] [--attention] [--out DIR] <build-dir> <reference-dir>
```

| Stage | Runs | Passes when |
|---|---|---|
| `build` (not in the default list) | `cmake --build <build-dir>` | it builds |
| `shards` | `tools/tp2/check_shard_sources.py` on the source tree, no GPU | no upstream definition outside an anonymous namespace in a source a shard re-includes, no renamed identifier in a header the shard's prelude does not include first |
| `ctest` | every CTest except the tp1 27B tests that do not fit one 16 GB board and, without `--attention`, `ninfer_softmax_attention_test` (~12 min); `NINFER_TEST_ARTIFACT` is the `_df2` artifact | all pass (skips allowed) |
| `golden` | `tools/golden/record.sh` on the synthetic tp1 model | ids equal the reference's |
| `ppl` | `ninfer-perplexity --tp 2 --quick` on the QUASAR artifact, INT8 KV, 65536/32768 and 4096/2048 | the per-source table equals the reference's to every printed digit |
| `greedy` | `ninfer-serve` with the production flags (MTP3, `--lm-head-draft`, C=1), the reference's 60 prompts, T=0, 128 tokens | texts identical; ms/round (median of decode seconds per MTP round) PASS within ±1 % of the reference, WARN beyond |
| `dflash2` | `ninfer-serve` DFlash2 K=7 `--lm-head-draft` on the `_df2` artifact, first 10 prompts | as `greedy` |

`--record` writes the reference directory from the given build (it needs `GATE_PROMPTS`, a JSONL
of `{"id","text"}`, copied into the reference). Every path has a `GATE_*` override
(`GATE_ARTIFACT`, `GATE_DF2_ARTIFACT`, `GATE_TEST_ARTIFACT`, `GATE_GOLDEN_MODEL`, `GATE_CORPUS`,
`GATE_DEVICES`, `GATE_PORT`, `GATE_SERVE_FLAGS`, `GATE_DF2_FLAGS`, `GATE_SRC`); the defaults are the
development workstation's. The script takes no GPU lease itself: wrap it
(`gpu-lease run ricerca --note ... -- tools/tp2/gate.sh ...`). Output and `summary.txt` go to
`<build-dir>/gate/<timestamp>/` or `--out`. About 13 minutes on two RTX 5070 Ti.

A WARN on ms/round is not a verdict: rerun the two builds alternated (A B B A) before deciding.

### The recorded reference

`reference/` is the reference the gate compares against on the development pair (two RTX 5070 Ti
without P2P, eco clocks at 2085 MHz, nvcc 13.2, `sm_120a`, the default `GATE_*` artifacts):
`tools/tp2/gate.sh <build> tools/tp2/reference`. It holds only what the stages compare (the
prompts, the texts and request logs of `greedy` and `dflash2`, the perplexity tables, the golden
ids) and the recording summaries. `ctest`, `golden`, `ppl` and `greedy` were recorded from `main`
169514ea (v0.3.0; unchanged and bit-identical since, `summary-main-169514ea.txt`); `dflash2` was
recorded again from v0.4.0 (60c47e45, the split DFlash2 drafter: same ten texts and acceptance,
18.392 ms/round against 19.952 with the drafter on rank 0, `summary.txt`). Another board has
other texts: record its own reference (`--record`) rather than comparing against this one.
