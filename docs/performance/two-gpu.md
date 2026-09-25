# Two-GPU mode: Qwen3.8-27B on 2× RTX 5070 Ti vs the published RTX 5090 runs

[Performance index](../performance.md) · [Methodology](methodology.md) · [Qwen3.8-27B (RTX 5090)](qwen3.8-27b.md)

Measured 2026-09-24/25 with upstream's own serving benchmark (`tools/bench/run_serve_corpus.py` and
`run_serve_concurrency.py`: fixed corpus, five seeds, stochastic sampling T 0.6 / top-p 0.95 / top-k 20 /
presence 1.0, INT8 group-64 KV, 1,024-token prefill chunk, prefix reuse off, one persistent server per
point). The only change is a wrapper that adds `--tp 2 --devices 0,1` to every `ninfer-serve` command.
The RTX 5090 column is upstream's published `nvfp4` profile ([runs N0, N3, NS, ND](qwen3.8-27b.md#scope-and-run-records),
2026-08-17 and 2026-09-06, CUDA 13.3, official artifact); it was not re-measured here, and this fork is
built on a later upstream commit (`bace20dc`), so a fresh 5090 run could move that column.

**Setup.** 2× RTX 5070 Ti 16 GB (PCIe 5.0 x8 + x8, no peer access: the captured all-reduces go through
the pinned-host mailbox), Ryzen 9 9900X, Linux, CUDA 13.1, **uncapped clocks** (3.1-3.2 GHz under load;
the README's "Measured" figures are at clocks capped to 2.1 GHz). Fork at `d24bffd2`.

Two weight sets on the pair:

- **official**: `qwen3_8_27b_nvfp4.ninfer`, the artifact of the 5090 runs (20.9 GiB: NVFP4 MLP in
  layers 0-55, FP8 elsewhere);
- **QUASAR**: [`qwen3_8_27b_quasar_nvfp4.ninfer`](../../model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md),
  the QUASAR-QAT checkpoint imported as encoded (every layer projection NVFP4, 17.0 GiB).

Rows with the official weights on both sides are the like-for-like hardware comparison. QUASAR rows
add the effect of lighter weights on the same pair; the 5090 was not run with QUASAR.

## Decode saturation (NS: 16,384-token context, one 8,192-token generation per active request, MTP3)

| C | RTX 5090 | pair, official | share | pair, QUASAR | share |
|--:|--:|--:|--:|--:|--:|
| 1 | 143.8 | 122.8 | 85% | 136.9 | 95% |
| 2 | 267.6 | 210.7 | 79% | 248.0 | 93% |
| 4 | 461.1 | 367.6 | 80% | 438.9 | 95% |
| 8 | 766.6 | 620.9 | 81% | 705.0 | 92% |

Steady decode tok/s over the full wave; MTP acceptance 57-60% everywhere.

## Context-length profile, no speculation (N0: NIAH fixtures, five seeds, 128 output tokens)

| prompt tokens | metric | RTX 5090 | pair, official | share | pair, QUASAR | share |
|--:|---|--:|--:|--:|--:|--:|
| 7,680 | prefill tok/s | 8,340.4 | 4,820.0 | 58% | 5,932.7 | 71% |
| | server TTFT s | 0.93 | 1.60 | | 1.30 | |
| | decode tok/s | 71.2 | 68.7 | 96% | 81.3 | 114% |
| 64,512 | prefill tok/s | 5,297.9 | 3,621.5 | 68% | 4,217.6 | 80% |
| | server TTFT s | 12.3 | 17.8 | | 15.3 | |
| | decode tok/s | 65.7 | 62.9 | 96% | 73.3 | 112% |
| 130,048 | prefill tok/s | 3,544.7 | 2,782.2 | 78% | 3,121.5 | 88% |
| | server TTFT s | 36.9 | 46.8 | | 41.7 | |
| | decode tok/s | 59.6 | 57.5 | 96% | 66.2 | 111% |
| 260,096 | prefill tok/s | 2,203.1 | 1,901.6 | 86% | 2,053.3 | 93% |
| | server TTFT s | 118.4 | 136.9 | | 126.8 | |
| | decode tok/s | 52.9 | 49.5 | 94% | 55.8 | 105% |

Plain decode is memory-bound: the pair has the 5090's bandwidth (2 × 896 GB/s) and each board reads
half the weights, so it lands at 94-96% with the same weights and above the 5090 with the 19% lighter
QUASAR weights. Prefill is compute-bound and pays the tensor-parallel exchange once per 1,024-token
chunk, a fixed cost that shrinks relative to attention as the prompt grows: 58% at 7.7k, 86% at 260k.

## Corpus makespan (N3: 75 fixed requests, MTP3, 131,072-token context, `--kv-capacity auto`)

| C | RTX 5090: makespan s · corpus decode tok/s · acceptance | pair, QUASAR | share | pair, official |
|--:|---|---|--:|---|
| 1 | 4,670 · 161.1 · 60.8% | 4,900 · 152.0 · 58.5% | 94% | 5,158 · 139.7 · 61.6% (87%) |
| 2 | 2,511 · 294.7 · 59.2% | 2,702 · 264.6 · 58.0% | 90% | — |
| 4 | 1,648 · 432.9 · 58.0% | 1,679 · 417.9 · 58.7% | 97% | — |
| 8 | 2,165 · 334.2 · 57.6% | 1,521 · 489.5 · 59.7% | 146% | — |

At C=8 the 5090 is memory-bound (its auto KV capacity resolved to 187,712 tokens and the average batch
fell to 2.36) while the pair keeps 282,112 tokens of KV across two boards (average batch 4.33). The
pair's C=8 point was not run with the official weights.

## Per-request decode phase by category (C=1 corpus, `(completion − 1) / decode seconds`, mean over 15 requests; AIME rows 5)

| fixture group | RTX 5090 MTP3 | pair QUASAR MTP3 | pair official MTP3 | RTX 5090 DFlash2 K=7 | pair QUASAR DFlash2 K=7 |
|---|--:|--:|--:|--:|--:|
| code | 194.3 | 186.0 ± 12.1 | 168.6 | 265.5 | 231.3 ± 26.6 |
| story | 126.1 | 122.7 ± 10.4 | 108.8 | 121.3 | 104.7 ± 25.1 |
| translation | 192.3 | 186.8 ± 13.4 | 165.8 | 255.9 | 212.8 ± 49.5 |
| structured | 219.8 | 211.1 ± 10.7 | 188.7 | 356.8 | 322.5 ± 38.2 |
| AIME26 #01 | 195.2 | 188.4 ± 4.8 | 168.3 | 321.1 | 288.3 ± 15.6 |
| AIME26 #15 (65k output) | 151.4 | 143.6 ± 3.6 | 131.6 | 183.4 | 159.3 ± 5.1 |
| AIME26 #30 | 167.5 | 155.3 ± 1.2 | 145.6 | 199.6 | 172.9 ± 2.5 |

DFlash2 on the pair uses a QUASAR artifact with the `z-lab/Qwen3.8-27B-DFlash2` drafter added; that
build is not published (it leaves 0.44 GiB of headroom on rank 0 at 196,608 tokens with Vision). The
DFlash2 K=7 corpus (ND, C=1): pair 166.6 tok/s over 4,274 s vs 5090 192.5 over 3,612 s (87%), acceptance
37.0% on both.

## Against llama.cpp on the same two boards (own harness, uncapped clocks)

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
python3 tools/bench/run_serve_concurrency.py --serve ./serve-tp2.sh --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --sampling stochastic --suite decode-saturation --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto --output profiles/bench/tp2_decode_saturation
python3 tools/bench/run_serve_corpus.py --serve ./serve-tp2.sh --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic --output profiles/bench/tp2_context_profile
python3 tools/bench/run_serve_concurrency.py --serve ./serve-tp2.sh --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto --output profiles/bench/tp2_corpus
```

`--lm-head-draft` (added by the runner for MTP3) works at `--tp 2`. The runners need Python 3.11+ and
the standard library only.
