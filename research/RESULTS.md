# D-window results (2x RTX 5060 Ti 16 GB, sm_120a, 36 SM, no P2P) - 2026-10-18

Build: ada-tree @ 95884c20 (= d01dbcf7 + d4050f1e + 95884c20), build-120a-cu13, CUDA 13, vision ON.
Primary artifact for the product requirement (NVFP4 weights + NVFP4 KV): W4A4 23,773,675,264 B
(/tank/models/lynn/NVFP4-NInfer/W4A4/...), sha256 3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808.

## Gate (pre-window)
- audit51 d01dbcf7: build rc=0, sub 28/28 425.90 s, cluster 174/174 1056.48 s (138P/36S/0F),
  qual 1/1 757.47 s, dtypes nvfp4/int8/bf16 rc=0 -> tag v-tp2-audit51-...

## D1 capability (W4A4, int8 KV unless said)
- serve tp2 + MTP3 + --lm-head-draft + --vision --vision-device 0: health 200; /v1/models ok.
- text 82 tok / 1.506 s (54.5 tok/s single request, includes prefill).
- media prompt -> HTTP 500 "multimodal prefill is not implemented at tensor-parallel width 2".
- accepted_lengths: present, 23 steps = 23 rounds, hist [2,1,0,20], acc 61, fb 0, 0 mismatches.
- perplexity --tp 2 (first reachable): int8 PPL 4.259926 / nvfp4 PPL 4.267108, both 1580.7 tok/s.
- nvfp4 head-local parity: ninfer_attention_headlocal_test rc=0, "OK head-local attention".

## D2 capacity / allowance
- per-rank weights (serve): r0 10.7 GiB (sharded 9.27, replicated 1.19, local 282.5 MiB), r1 10.5 GiB.
- scoring phase: both 10.1 GiB (sharded 8.90, replicated 1.19); used-memory delta r0-r1 = 728 MiB.
- capacity int8 KV @32768: pages 512/512, runtime 1.74 GiB, free 2.87 GiB.
- capacity int8 KV @196608 (vision+MTP3): pages 3072/3072, runtime 4.48 GiB, free 132.7 MiB -> fits.
- capacity int8 KV @262144: startup fails, needs 5,984,344,833 B, 4,947,902,464 B available.
- capacity nvfp4 KV @32768: pages 512/512, runtime 796.6 MiB, free 4.10 GiB.
- graph allowance observed: ordinary 0.0/0.0 (allow 8), MTP 2.0/2.0 (allow 8), DFlash2 22.0/12.0 (allow 66 = 6x11).
- shard TMA bound 1024: no shard bench in tree -> placeholder kept, registered as a gap.
- nvfp4 12/2 row group / key shift: structural (plan reads device SM count), no field to write back.

## D3 verbatim + GSM8K
- record baseline -> compare: greedy 60/60 identical (51.96 tok/s, accept 72.43%, ms/round 60.035,
  -0.09% vs ref); dflash2 10/10 identical (133.94 tok/s, accept 59.20%, ms/round 37.817, +0.09%).
- GSM8K 500 (int8 KV, thinking off, 8-shot, max 512): canonical 484/500 = 96.8%, numeric 96.8%,
  unmatched_gold 0, request_errors 0, digest 245dfab7...
- GSM8K 500 (nvfp4 KV): canonical 482/500 = 96.4%, numeric 96.4%, 0 errors, same digest.

## D4 speed
- ordinary 48-token sample: 39.0 tok/s (vision off).
- MTP3 int8 196608: 51.92 tok/s, accept 72.43%, ms/round 60.088 (record run).
- DFlash2 K7 int8 32768: 134.06 tok/s, accept 59.20%, ms/round 37.781.
- nvfp4 face record/compare: PASS; greedy 60/60 identical (52.17 tok/s, accept 72.18%, 59.853 ms/round),
  dflash2 10/10 identical (140.38 tok/s, accept 62.09%, 37.531 ms/round).
- nvfp4 sweep (60 prompts): K2 45.10 tok/s/80.69%/57.376 ms; K3 52.39/72.18%/59.594;
  K4 54.85/64.76%/64.252; K5 56.96/57.09%/65.767 (fastest). MTP draft-tokens max is 5.

## Fixes landed this window
- 95884c20: MTP Finish phase guard + DFlash2 verify card TpExecution (both two-device tests green).
- d4050f1e: sm89 configure gate (stale test names) + pkg-config operator spacing.
- Open regression (registered): sm89 FP8 A8 TMA schedules, ~15 TUs, .tile requires sm_90.
- Open gap (registered): NF4 artifact tp2 blocked (q6 output head has no half-vocab shape).
