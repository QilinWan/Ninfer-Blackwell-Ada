# Two-GPU mode: Qwen3.8-27B on 2× RTX 5070 Ti vs the published RTX 5090 runs

[Performance index](../performance.md) · [Methodology](methodology.md) · [Qwen3.8-27B (RTX 5090)](qwen3.8-27b.md)

Measured 2026-09-28 on **v0.2.2** (`44a58463`: v0.2.1 plus the pipelined mailbox exchange kernel) with
upstream's own serving benchmark (`tools/bench/run_serve_corpus.py` and `run_serve_concurrency.py`:
fixed corpus, stochastic sampling T 0.6 / top-p 0.95 / top-k 20 / presence 1.0, INT8 group-64 KV,
1,024-token prefill chunk, prefix reuse off, one persistent server per point, MTP3 with
`--lm-head-draft`). Two departures from upstream's runs: a wrapper adds `--tp 2 --devices 0,1` to every
`ninfer-serve` command, and every fixture ran with **three seeds instead of five** (the first three of
the runner's `SEEDS`): 45 corpus requests instead of 75 and three NIAH samples per depth. Corpus
makespans are therefore not comparable with the 5090's; corpus decode rates and per-request rates are
(on the earlier run, keeping only these three seeds moved the C=1 corpus rates by −2.4% to +0.3%).
The RTX 5090 column is upstream's published `nvfp4` profile ([runs N0, N3, NS, ND](qwen3.8-27b.md#scope-and-run-records),
2026-08-17 and 2026-09-06, CUDA 13.3, official artifact); it was not re-measured here.

**Setup.** 2× RTX 5070 Ti 16 GB (PCIe 5.0 x8 + x8, no peer access: the captured all-reduces go through
the pinned-host mailbox), Ryzen 9 9900X, Linux, CUDA 13.1, driver 595.91. **Core clocks locked at
2,100 MHz** (`nvidia-smi -lgc 2100`, the pair's everyday operating point): 2.03-2.09 GHz under load,
memory 13,801 MHz. The [earlier run](#earlier-run-2026-09-2425-v01x-uncapped-clocks) had uncapped
clocks, about 2.9 / 2.8 GHz under load on the two boards; on the same binary the lock costs
17-18% of decode and 14-22% of prefill ([below](#what-changed-since-the-earlier-run)), so the
shares against the 5090 here are lower than there for that reason, not because of the code.

Two weight sets on the pair:

- **official**: `qwen3_8_27b_nvfp4.ninfer`, the artifact of the 5090 runs (20.9 GiB: NVFP4 MLP in
  layers 0-55, FP8 elsewhere);
- **QUASAR**: [`qwen3_8_27b_quasar_nvfp4.ninfer`](../../model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md),
  the QUASAR-QAT checkpoint imported as encoded (every layer projection NVFP4, 17.0 GiB).

Rows with the official weights on both sides are the like-for-like hardware comparison. QUASAR rows
add the effect of lighter weights on the same pair; the 5090 was not run with QUASAR. On v0.2.2 the
official weights were re-measured only for the headline points (plain decode and prefill at 7,680
tokens, MTP3 corpus at C=1); their other points are in the earlier run.

## Decode saturation (NS: 16,384-token context, one 8,192-token generation per active request, MTP3)

| C | RTX 5090 | pair, QUASAR | share |
|--:|--:|--:|--:|
| 1 | 143.8 | 123.3 | 86% |
| 2 | 267.6 | 206.3 | 77% |
| 4 | 461.1 | 358.5 | 78% |
| 8 | 766.6 | 599.7 | 78% |

Steady decode tok/s over the full wave, QUASAR weights; MTP acceptance 45-46%. C=1 is the only
point where the pipelined mailbox kernel acts (see [below](#what-changed-since-the-earlier-run)):
batched rounds exceed the mailbox slot and keep the staged path.

## Context-length profile, no speculation (N0: NIAH fixtures, three seeds, 128 output tokens)

| prompt tokens | metric | RTX 5090 | pair, official | share | pair, QUASAR | share |
|--:|---|--:|--:|--:|--:|--:|
| 7,680 | prefill tok/s | 8,340.4 | 4,134.2 | 50% | 5,125.3 | 61% |
|  | server TTFT s | 0.93 | 1.86 |  | 1.50 |  |
|  | decode tok/s | 71.2 | 58.6 | 82% | 68.4 | 96% |
| 64,512 | prefill tok/s | 5,297.9 | — | — | 3,462.3 | 65% |
|  | server TTFT s | 12.28 | — |  | 18.66 |  |
|  | decode tok/s | 65.7 | — | — | 61.2 | 93% |
| 130,048 | prefill tok/s | 3,544.7 | — | — | 2,475.3 | 70% |
|  | server TTFT s | 36.85 | — |  | 52.63 |  |
|  | decode tok/s | 59.6 | — | — | 55.3 | 93% |
| 260,096 | prefill tok/s | 2,203.1 | — | — | 1,604.2 | 73% |
|  | server TTFT s | 118.35 | — |  | 162.30 |  |
|  | decode tok/s | 52.9 | — | — | 46.5 | 88% |

Plain decode is memory-bound: the pair has the 5090's bandwidth (2 × 896 GB/s) and each board reads
half the weights, and the clock lock touches it less than prefill. Prefill is compute-bound, pays
the lock in full and pays the tensor-parallel exchange once per 1,024-token chunk, a fixed cost that
shrinks relative to attention as the prompt grows.

## Corpus makespan (N3: fixed corpus, MTP3, 131,072-token context, `--kv-capacity auto`)

| C | RTX 5090 (75 requests): makespan s · corpus decode tok/s · acceptance | pair, QUASAR (45 requests): makespan s · corpus decode tok/s · acceptance · avg batch | share (tok/s) |
|--:|---|---|--:|
| 1 | 4,670 · 161.1 · 60.8% | 3,090 · 139.6 · 58.3% · 1.00 | 87% |
| 2 | 2,511 · 294.7 · 59.2% | 1,812 · 221.8 · 60.3% · 1.85 | 75% |
| 4 | 1,648 · 432.9 · 58.0% | 1,122 · 350.0 · 59.0% · 3.47 | 81% |
| 8 | 2,165 · 334.2 · 57.6% | 1,067 · 401.3 · 59.0% · 4.15 | 120% |
| 1, official | 4,670 · 161.1 · 60.8% | 3,648 · 123.4 · 58.7% · 1.00 | 77% |

The pair's makespans cover 45 requests, the 5090's 75; compare the rates. At C=8 the 5090 is
memory-bound (its auto KV capacity resolved to 187,712 tokens and the average batch fell to 2.36)
while the pair keeps 282,112 tokens of KV across two boards (average batch 4.15). The pair's auto KV
capacity resolved to 131,072 / 262,144 / 302,784 / 282,112 tokens at C=1 / 2 / 4 / 8, as in the
earlier run.

## Per-request decode phase by category (C=1 corpus, `(completion − 1) / decode seconds`)

Mean ± population standard deviation over 9 requests per category, 3 per AIME row (the 5090: 15 and
5).

| fixture group | RTX 5090 MTP3 | pair QUASAR MTP3 | pair official MTP3 | RTX 5090 DFlash2 K=7 | pair QUASAR DFlash2 K=7 |
|---|--:|--:|--:|--:|--:|
| code | 194.3 | 170.6 ± 10.5 | 152.0 | 265.5 | 228.7 ± 13.6 |
| story | 126.1 | 113.6 ± 9.5 | 99.3 | 121.3 | 105.1 ± 24.1 |
| translation | 192.3 | 171.1 ± 12.0 | 148.7 | 255.9 | 215.8 ± 40.5 |
| structured | 219.8 | 198.1 ± 6.9 | 172.4 | 356.8 | 319.5 ± 39.6 |
| AIME26 #01 | 195.2 | 177.5 ± 2.3 | 153.8 | 321.1 | 270.4 ± 10.6 |
| AIME26 #15 (65k output) | 151.4 | 132.8 ± 3.0 | 118.7 | 183.4 | 148.3 ± 1.2 |
| AIME26 #30 | 167.5 | 138.6 ± 0.7 | 122.5 | 199.6 | 165.4 ± 5.1 |

DFlash2 on the pair uses a QUASAR artifact with the `z-lab/Qwen3.8-27B-DFlash2` drafter added; that
build is not published (it leaves 0.44 GiB of headroom on rank 0 at 196,608 tokens with Vision).
The DFlash2 K=7 corpus (ND, C=1): pair 160.4 tok/s over 2,814 s (45 requests) vs 5090 192.5 over
3,612 s (75), 83%; acceptance 36.6% vs 37.0%. Fastest single requests on the pair: 207.7 tok/s with
MTP3 and 370.6 tok/s with DFlash2, both structured output.

## What changed since the earlier run

Same pair, same 2,100 MHz lock, same three seeds, QUASAR weights, measured the same night: the
earlier run's binary (v0.1.x `d24bffd2`), v0.2.1 and v0.2.2, and v0.2.2 with
`NINFER_TP_MAILBOX_LEGACY=1` (the original exchange kernel on the same binary; the combine arithmetic
is the same, so the tokens are identical).

| metric | v0.1.x `d24bffd2` | v0.2.1 `344f008d` | v0.2.2 `44a58463` | v0.2.2, legacy mailbox kernel |
|---|--:|--:|--:|--:|
| MTP3 decode saturation, C=1, tok/s | 112.3 | 113.4 | 123.3 | — |
| MTP3 decode saturation, C=8, tok/s | 582.1 | 598.5 | 599.7 | — |
| plain decode at 7,680 tokens, tok/s | 67.5 | 65.6 | 68.4 | 65.7 |
| prefill at 7,680 tokens, tok/s | 5,094 | 5,213 | 5,125 | 5,236 |
| plain decode at 260,096 tokens, tok/s | 46.1 | 45.2 | 46.5 | — |
| prefill at 260,096 tokens, tok/s | 1,599 | 1,610 | 1,604 | — |
| MTP3 corpus C=1, tok/s | — | 127.5 | 139.6 | 128.2 |
| MTP3 corpus C=1, ms per round | — | 21.53 | 19.67 | 21.42 |

- **v0.2.1 against v0.1.x**: prefill 0.7-2.3% faster, MTP3 steady decode within 1% at C=1-2 and
  within 4% at C=4-8 (single waves; the gaps follow the wave's acceptance), plain decode 2.1-2.9%
  slower at every depth. The upstream merge (unified linear templates, two-stage GDN) changed the numerics: the
  saturation wave's MTP acceptance moved from 46.6% to 45.2% at C=1; on the corpus it stays at 58.3%.
- **v0.2.2 against v0.2.1**: the pipelined mailbox kernel cuts the MTP3 round from 21.4 to 19.7 ms
  (−8.2% against the legacy kernel on the same binary, identical tokens), which gives +8.9% corpus
  decode at C=1 and +4.2% plain decode at 7.7k against the legacy kernel, and against v0.2.1 +8.8%
  steady decode at C=1 and +2.9 to +4.4% plain decode at every depth. Batched rounds (C ≥ 2) exceed the
  40 KiB mailbox slot and keep the staged path, so C=2-8 do not move; prefill always runs eager on the
  staged path, and its 1-2% spread between servers is noise (the three v0.2.2 samples at 7.7k:
  5,209 / 5,132 / 5,034).
- **The clock lock**, same v0.1.x binary, uncapped (2026-09-24) against 2,100 MHz: plain decode
  −17.0 to −17.4%, MTP3 steady decode −17.4 to −18.0% at C=1 and C=8, prefill −14% at 7.7k growing to
  −22% at 260k. v0.2.2 under the lock stays 8.4% under the earlier uncapped MTP3 corpus at C=1 (139.6
  vs 152.4 on the same three seeds; official weights 123.4 vs 136.4, −9.5%) and 3.1% under its DFlash2
  corpus (160.4 vs 165.5, 22.2 vs 21.5 ms per round).

## Earlier run (2026-09-24/25, v0.1.x, uncapped clocks)

Fork at `d24bffd2` (before the upstream merge), uncapped clocks (2.9 / 2.8 GHz under load), five
seeds, 75 corpus requests; same runners and wrapper. Shares against the same 5090 column.

| metric | RTX 5090 | pair, official | share | pair, QUASAR | share |
|---|--:|--:|--:|--:|--:|
| NS steady decode, C=1 / 2 / 4 / 8, tok/s | 143.8 / 267.6 / 461.1 / 766.6 | 122.8 / 210.7 / 367.6 / 620.9 | 79-85% | 136.9 / 248.0 / 438.9 / 705.0 | 92-95% |
| N0 plain decode at 7,680 / 64,512 / 130,048 / 260,096 tokens, tok/s | 71.2 / 65.7 / 59.6 / 52.9 | 68.7 / 62.9 / 57.5 / 49.5 | 94-96% | 81.3 / 73.3 / 66.2 / 55.8 | 105-114% |
| N0 prefill at the same depths, tok/s | 8,340 / 5,298 / 3,545 / 2,203 | 4,820 / 3,622 / 2,782 / 1,902 | 58-86% | 5,933 / 4,218 / 3,122 / 2,053 | 71-93% |
| N0 server TTFT at the same depths, s | 0.93 / 12.3 / 36.9 / 118.4 | 1.60 / 17.8 / 46.8 / 136.9 | | 1.30 / 15.3 / 41.7 / 126.8 | |
| N3 corpus decode, C=1 / 2 / 4 / 8, tok/s | 161.1 / 294.7 / 432.9 / 334.2 | 139.7 (C=1) | 87% | 152.0 / 264.6 / 417.9 / 489.5 | 90-97%, 146% at C=8 |
| N3 makespan (75 requests), C=1 / 2 / 4 / 8, s | 4,670 / 2,511 / 1,648 / 2,165 | 5,158 (C=1) | | 4,900 / 2,702 / 1,679 / 1,521 | |
| ND DFlash2 K=7 corpus, C=1, tok/s (acceptance) | 192.5 (37.0%) | — | | 166.6 (37.0%) | 87% |
| decode phase by category, MTP3: code / story / translation / structured / AIME #01 / #15 / #30 | 194.3 / 126.1 / 192.3 / 219.8 / 195.2 / 151.4 / 167.5 | 168.6 / 108.8 / 165.8 / 188.7 / 168.3 / 131.6 / 145.6 | 86-87% | 186.0 / 122.7 / 186.8 / 211.1 / 188.4 / 143.6 / 155.3 | 93-97% |
| the same, DFlash2 K=7 | 265.5 / 121.3 / 255.9 / 356.8 / 321.1 / 183.4 / 199.6 | — | | 231.3 / 104.7 / 212.8 / 322.5 / 288.3 / 159.3 / 172.9 | 83-90% |

At C=8 the 5090 is memory-bound (its auto KV capacity resolved to 187,712 tokens and the average batch
fell to 2.36) while the pair kept 282,112 tokens of KV across two boards (average batch 4.33).

### Against llama.cpp on the same two boards (own harness, earlier run, uncapped clocks)

llama.cpp `67672dc5b` with `-sm tensor`, UD-Q4_K_XL, KV q8_0 and the MTP draft model (`--spec-draft-n-max 2`),
one request at a time; NInfer with INT8 KV and MTP3. The context ladder uses repetitive filler, which
inflates speculative acceptance on both sides.

| load | llama.cpp | NInfer tp2, official | NInfer tp2, QUASAR |
|---|--:|--:|--:|
| decode tok/s at 0 / 16k / 64k / 123k / 184k context | 119.7 / 125.8 / 98.4 / 80.4 / 64.0 | 135.4 / 141.6 / 131.3 / 115.7 / 112.2 | 145.4 / 148.1 / 137.5 / 125.9 / 134.4 |
| cold prefill TTFT at 16k / 64k / 121k / 184k | 7.5 / 34.0 / 74.5 / 129.1 s | 3.7 / 18.4 / 43.5 / 79.8 s | 3.0 / 15.9 / 38.8 / 72.7 s |
| short prompts, decode tok/s (prose / code / math / list) | 117.5 / 154.7 / 148.4 / 126.0 | 110.5 / 164.3 / 156.4 / 117.1 | 120.4 / 184.6 / 171.6 / 137.9 |
| 16-turn agent session to 93.6k tokens: wall · mean TTFT · GPU energy | 82.8 s · 3.62 s · 39.0 kJ | 49.7 s · 1.90 s · 22.3 kJ | 45.2 s · 1.68 s · 19.6 kJ |

## Quality

QUASAR weights, NInfer tp2 vs vLLM 0.30, paired per item (exact McNemar): GSM8K 0.985 vs 0.975
(p 0.69), MMLU-Pro (308) 0.789 vs 0.802 (p 0.48), IFEval (200) 0.870 vs 0.880 (p 0.83). COMET on
FLORES-200 it↔en 0.8850 / 0.8910 against 0.8847 / 0.8918 for the official weights on vLLM. Details in
the [model card](../../model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md).

## Reproduction

```bash
cat > serve-tp2.sh <<'EOF'
#!/usr/bin/env bash
# drop the runner's single-GPU --device N, add the two-GPU options
args=(); skip=0
for a in "$@"; do
  if [ $skip = 1 ]; then skip=0; continue; fi
  case "$a" in --device) skip=1; continue;; esac
  args+=("$a")
done
exec ./build/apps/ninfer-serve "${args[@]}" --tp 2 --devices 0,1
EOF
chmod +x serve-tp2.sh
sudo nvidia-smi -lgc 2100    # the clock lock of these tables (-rgc for the earlier run's uncapped clocks)
python3 tools/bench/run_serve_concurrency.py --serve ./serve-tp2.sh --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --sampling stochastic --suite decode-saturation --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto --output profiles/bench/tp2_decode_saturation
python3 tools/bench/run_serve_corpus.py --serve ./serve-tp2.sh --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic --output profiles/bench/tp2_context_profile
python3 tools/bench/run_serve_concurrency.py --serve ./serve-tp2.sh --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto --output profiles/bench/tp2_corpus
python3 tools/bench/run_serve_concurrency.py --serve ./serve-tp2.sh --artifact qwen3_8_27b_df2=out/qwen3_8_27b_quasar_nvfp4_df2.ninfer \
  --mode dflash2_7 --sampling stochastic --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto --prefill-chunk 1024 --output profiles/bench/tp2_dflash2
```

The three-seed runs import the runners and cut `SEEDS` before calling `main()`:

```python
import sys; sys.path.insert(0, ".")
from tools.bench import run_serve_corpus as corpus, run_serve_concurrency as conc
corpus.SEEDS = corpus.SEEDS[:3]
sys.exit(conc.main(sys.argv[1:]))   # or corpus.main(...) for the context profile
```

`--lm-head-draft` (added by the runner for MTP3) works at `--tp 2`. The runners need Python 3.11+ and
the standard library only.
