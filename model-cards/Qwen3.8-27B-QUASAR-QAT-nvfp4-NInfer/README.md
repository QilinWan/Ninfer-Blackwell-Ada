---
license: apache-2.0
base_model: QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - quantization-aware-training
  - blackwell
  - tensor-parallel
  - multi-gpu
  - dual-gpu
  - rtx-5070-ti
---

# Qwen3.8-27B QUASAR-QAT NVFP4 on two 16 GB GPUs — NInfer artifact (`qwen3_8_27b_quasar_nvfp4.ninfer`)

> **Runs Qwen3.8-27B on two 16 GB GPUs.** Verified on 2× RTX 5070 Ti (PCIe, no P2P) with tensor
> parallelism (`--tp 2`) in [ninfer-tp2](https://github.com/ValerioDolci/ninfer-tp2): 196,608-token context
> with Vision and MTP at 13,083 / 12,743 MiB per board; 262,144 tokens also fit.

An all-NVFP4 `.ninfer` v3 artifact of Qwen3.8-27B built from the
[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
checkpoint (quantization-aware training, every layer projection in NVFP4), for the
[two-GPU tensor-parallel fork of NInfer](https://github.com/ValerioDolci/ninfer-tp2). Compared with the
official `qwen3_8_27b_nvfp4.ninfer` (NVFP4 MLP in layers 0-55, FP8 elsewhere) it needs **17.0 GiB of
weights instead of 20.9**, which on two 16 GB boards buys either ~2 GiB per board or the full 262,144-token
context with Vision and 8 device state slots.

- **Weights:** 512 projections NVFP4 (imported as encoded by QUASAR: codes, block scales and global scale,
  no requantization), `lm_head` and embedding FP8 rows (the official method), GDN `a`/`b` projections
  decoded from QUASAR's NVFP4 to BF16 (the runtime wants them unquantized), norms as stored. Vision tower
  and MTP head are read from the BF16 copies inside the QUASAR checkpoint and quantized with the official
  `_optional` choices (Q4-Q8). Components: text, vision, MTP (the DFlash2 drafter is not included).
- **Source revision:** `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` at `15d2e47b`.
- **Size:** 17.4 GB on disk. At `--tp 2` with the production flags below: 13,083 / 12,743 MiB per board
  (official NVFP4 artifact: 15,035 / 14,695).

## Sources and licenses

| Source | Role | License |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | base model (architecture; the checkpoint below carries its vision and MTP weights) | Apache-2.0 |
| [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | all weights: NVFP4 projections (QAT) plus BF16 vision, MTP, head | Apache-2.0 |
| [Neroued/ninfer](https://github.com/Neroued/ninfer) | converter (unchanged in the fork), single-GPU runtime | Apache-2.0 |
| [ValerioDolci/ninfer-tp2](https://github.com/ValerioDolci/ninfer-tp2) | two-GPU runtime this artifact was verified with | Apache-2.0 |

This artifact is redistributed under Apache-2.0. All credit for the quantization goes to the QUASAR-QAT
authors; this card only documents the conversion and the measurements below.

**Prior art.** A QUASAR-QAT NInfer artifact was published first by
[MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer)
(2026-08-26, with the DFlash2 drafter, MTP and Vision, built with upstream's converter from the BF16 base
plus the QUASAR checkpoint). This one was built independently from the QUASAR checkpoint alone
(`import_encoded`, no BF16 copy), leaves the DFlash2 drafter out so that 196,608 tokens with Vision fit two
16 GB boards with MTP3, and is the artifact the two-GPU measurements above were taken with.

## Conversion

Recipe [`quasar_recipe.py`](quasar_recipe.py) (next to this card; at the root of the Hub repository). The
converter is upstream's (unchanged in the fork); running the result at `--tp 2` needs the fork at commit
`d24bffd2` or later for the NVFP4 split projections. The QUASAR checkpoint is both the `--model` and the
`quantized` source; no BF16 copy of Qwen3.8-27B is required. From the root of a `ninfer-tp2` checkout:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --recipe model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/quasar_recipe.py \
  --source quantized=/path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --device cpu \
  --out qwen3_8_27b_quasar_nvfp4.ninfer
```

With `--device cpu` the conversion takes about two minutes (0.75 GB RAM); the default is `cuda`.

### Variant: MTP layer in NVFP4 (not on the Hub)

`--recipe model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/quasar_recipe.py:configure_mtp_nvfp4`, same
command otherwise, stores the MTP layer's MLP gate/up, MLP down and attention output as NVFP4
(`nvfp4_mse` from QUASAR's BF16 MTP weights, 16-bit activations; `_mtp_nvfp4_layer` in
[`official_recipes.py`](../../tools/convert/official_recipes.py)). The attention input projection and
`mtp/input_projection` stay Q8; every other object is byte-identical to the artifact above. Result:
17,248,277,764 bytes (−142.5 MiB), SHA-256 `3363b1ab64c5693c8dcb0cb283b01bcb1a19de946846799e53bca7ac8f2533ac`
(2026-10-05). It runs on the same fork builds as the published artifact (the MTP halves use the NVFP4
linear routes of the target's MLP and attention output).

On 2× RTX 5070 Ti (eco clocks, MTP3, `--lm-head-draft`, 262,144 tokens, C=1, T=0) against the published
artifact: the MTP round is 1.9 % shorter, the outputs and the target perplexity are identical, and the
acceptance moves by −1.0…+0.7 points on short prompts and −1.7 points on agent prompts of 55–171K
tokens; decode is 1.5–2.9 % faster on code, math and English prose, 0.6 % on Italian prose and unchanged
on the long agent prompts. GSM8K (500 problems, 5-shot, thinking on, T=0, C=1): 483/500 on both
artifacts, the same answers on every problem (McNemar p = 1.000), with decode 186.6 → 190.1 tok/s.

## Provenance

Next to the artifact on the Hub: [`SHA256SUMS`](SHA256SUMS) (the artifact's digest — check a download with
`sha256sum -c SHA256SUMS`), [`artifact-manifest.json`](artifact-manifest.json) (digest, inventory, source
repositories at the exact revisions the weights were read from, converter and runtime revisions,
validation hardware) and [`qwen3_8_27b_quasar_nvfp4.ninfer.conversion.json`](qwen3_8_27b_quasar_nvfp4.ninfer.conversion.json),
the report the converter wrote (every object's method and sources, formats, timing).

- SHA-256 `808a0fa3bb5b3256aadef0d5c8264c106122d1e8483bd2cf74db4e1183cbb570`, 17,397,699,076 bytes,
  container v3, artifact id `1a09e96c3a16496aa60ce2895402bcd7`.
- Weights read from `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` at `15d2e47bffe5d8ad23928879f8f7d2f74909e259`
  (2026-09-16), the only tensor source.
- Converted on 2026-09-24 with upstream's `tools.convert` from this fork at `fa8c9e39` (converter unchanged
  from upstream v3); verified at `--tp 2` with the fork at `d24bffd2` on 2× RTX 5070 Ti, CUDA 13.1.1,
  driver 595.91.

## Serving (production flags on 2× RTX 5070 Ti 16 GB, no P2P)

```bash
ninfer-serve qwen3_8_27b_quasar_nvfp4.ninfer --tp 2 --devices 0,1 --kv-dtype int8 \
  --max-context 196608 --kv-capacity 196608 --device-state-slots 4 --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --vision --vision-device 0 --max-vision-tokens 4096
```

262,144 tokens with `--vision` and 8 device state slots also fit. Single GPU (`--tp 1`) should fit a 24 GB+
`sm_120a` board (untested).

## Measured quality (2026-09-24, this fork at `--tp 2`, flags above, one run per task, T = 0)

Paired per-item comparison against the same QUASAR weights served by vLLM 0.30 (exact McNemar):

| Task | NInfer tp2 | vLLM 0.30 | difference (95 % CI) | McNemar p |
|---|---:|---:|---:|---:|
| GSM8K (200, 5-shot, thinking) | 0.985 | 0.975 | +1.0 pt [−1.4, +3.4] | 0.69 |
| MMLU-Pro (308 = 22 × 14 categories, CoT, thinking) | 0.789 | 0.802 | −1.3 pt [−4.0, +1.4] | 0.48 |
| IFEval (200, prompt-level strict, thinking) | 0.870 | 0.880 | −1.0 pt [−5.6, +3.6] | 0.83 |

Against the official NVFP4 weights (`unsloth/Qwen3.8-27B-NVFP4`, vLLM): MMLU-Pro −1.0 pt (p 0.65),
IFEval ±0 (p 1.00), GSM8K +0.5 (p 1.00). COMET (wmt22-comet-da) on FLORES-200 devtest, 200 sentences per
direction, thinking off: it→en **0.8850** (official NVFP4 on vLLM: 0.8847), en→it **0.8910** (0.8918).
Synthetic long-context suite (needle + multi-hop with distractors, 12 trials per length, thinking off):
overall 1.000 at 8k and 0.667 at 126k — identical to the official NVFP4 artifact on the same runtime
at 126k, +1 trial at 8k. Multi-hop at 126k scores 0/4 with both weight sets. OpenAI-compatibility probe
(chat, streaming, thinking off, tool calls, vision, 32k outputs): 12/12.

Speed at `--tp 2` (eco clocks, 2,100 MHz): decode +9–18 % over the official NVFP4 artifact, prefill
4,892 t/s at 8k (official 3,920) and 2,521 t/s at 126k (2,238).

Method notes: lm-eval with `max_gen_toks 8192` and the answer read from `content` only; answers left
empty by an exhausted thinking budget count as wrong (15/308 on MMLU-Pro and 10/200 on IFEval here,
16/308 and 10/200 on vLLM). The pre-registered rule flagged a summed MMLU-Pro+IFEval shift beyond −2 pt
against vLLM as "undetermined": the measured sum is −2.3 pt (23 vs 17 discordant items over 508,
p ≈ 0.43), i.e. within paired noise but reported as such.

## Limits

- Built and verified only with this fork's two-GPU mode on sm_120 boards; the upstream single-GPU runtime
  should load it (same v3 format) but has not been tested here.
- No DFlash2 component. A variant with the `z-lab/Qwen3.8-27B-DFlash2` drafter exists and starts at
  196,608 tokens with Vision on two 16 GB boards, but is not published: it wins only on code/math
  (+10–14 %) and leaves 0.44 GiB of headroom on rank 0.
- One sample per task; the paired differences above are within noise, they are not evidence that QUASAR
  is better or worse than the official weights.
