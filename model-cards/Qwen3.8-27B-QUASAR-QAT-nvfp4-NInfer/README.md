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
---

# Qwen3.8-27B QUASAR-QAT NVFP4 — NInfer artifact (`qwen3_8_27b_quasar_nvfp4.ninfer`)

An all-NVFP4 `.ninfer` v3 artifact of Qwen3.8-27B built from the
[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
checkpoint (quantization-aware training, every layer projection in NVFP4), for the
[two-GPU tensor-parallel fork of NInfer](https://github.com/ValerioDolci/ninfer-tp2). Compared with the
official `qwen3_8_27b_nvfp4.ninfer` (NVFP4 MLP, FP8 attention and GDN projections) it needs **17.0 GiB of
weights instead of 20.9**, which on two 16 GB boards buys either ~2 GiB per board or the full 262,144-token
context with Vision and 8 device state slots.

- **Weights:** 512 projections NVFP4 (imported as encoded by QUASAR: codes, block scales and global scale,
  no requantization), `lm_head` and embedding FP8 rows (the official method), GDN `a`/`b` projections and
  norms BF16 (as stored). Components: text, vision, MTP (the DFlash2 drafter is not included).
- **Size:** 17.4 GB on disk. At `--tp 2` with the production flags below: 13,083 / 12,743 MiB per board
  (official NVFP4 artifact: 15,035 / 14,695).

## Sources and licenses

| Source | Role | License |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | base model (architecture, vision, MTP head) | Apache-2.0 |
| [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | NVFP4 weights (QAT) | Apache-2.0 |
| [Neroued/ninfer](https://github.com/Neroued/ninfer) | converter, runtime | Apache-2.0 |

This artifact is redistributed under Apache-2.0. All credit for the quantization goes to the QUASAR-QAT
authors; this card only documents the conversion and the measurements below.

## Conversion

Recipe [`quasar_recipe.py`](quasar_recipe.py) (in this directory), run with the converter of this fork
(commit `d24bffd2` or later — the NVFP4 split projections are needed at `--tp 2`). The QUASAR checkpoint is
both the `--model` and the `quantized` source; no BF16 copy of Qwen3.8-27B is required:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --recipe model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/quasar_recipe.py \
  --source quantized=/path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --out qwen3_8_27b_quasar_nvfp4.ninfer
```

The conversion takes about two minutes on a CPU (0.75 GB RAM).

## Serving (production flags on 2× RTX 5070 Ti 16 GB, no P2P)

```bash
ninfer-serve qwen3_8_27b_quasar_nvfp4.ninfer --tp 2 --devices 0,1 --kv-dtype int8 \
  --max-context 196608 --kv-capacity 196608 --device-state-slots 4 --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --vision --vision-device 0 --max-vision-tokens 4096
```

262,144 tokens with `--vision` and 8 device state slots also fit. Single GPU (`--tp 1`) needs a 24 GB+ board.

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
4,892 t/s at 8k (official 3,920) and 2,521 t/s at 126k (2,238); MTP3 acceptance 94 % over the lm-eval run.

## Limits

- Built and verified only with this fork's two-GPU mode on sm_120 boards; the upstream single-GPU runtime
  should load it (same v3 format) but has not been tested here.
- No DFlash2 component. A variant with the `z-lab/Qwen3.8-27B-DFlash2` drafter exists and starts at
  196,608 tokens with Vision on two 16 GB boards, but is not published: it wins only on code/math
  (+10–14 %) and leaves 0.44 GiB of headroom on rank 0.
- One sample per task; the paired differences above are within noise, they are not evidence that QUASAR
  is better or worse than the official weights.
