# 2026-09-30 (tune/attn-12x2) — cases 1 and 2 identical, case 3 moves with the INT8 split count

| | |
|---|---|
| upstream | `Neroued/ninfer` at `d44ab584`: the `upstream-d44ab584` ids of the `2026-09-30` record, not re-run |
| fork | branch `tune/attn-12x2`: recorded at `4b9c098e` (INT8 causal attention budgets its split-KV waves on the device's own SMs), re-run at `21ad2f4a` (splits as the slowest grid axis) with identical ids |
| artifact | the same synthetic two-layer model as `2026-09-27`, md5 `235749fafca92e11b760f1bfc7b8171a` (checked before the run) |
| hardware | one RTX 5070 Ti of the pair (device 0, 70 SMs), CUDA 13.2 (nvcc V13.2.86), driver 595.91.07 |

Cases 1 and 2 (BF16 KV) are identical to upstream `d44ab584`. Case 3 (INT8 KV, 6,144 prompt
tokens) diverges at generated token 100 and stops at 124 instead of 118. This is an intended tp 1
plan change, not a regression: upstream sizes the INT8 decode split for 170 SMs on every device,
so a 6,144-6,272 key decode row runs 48-49 KV splits (key granularity 128); on this 70-SM device
the fork now runs 35 (budget 2 * 70 CTAs over 4 KV heads). The merge order of the partials changes,
and token 100 of this case is a near tie that flips. On an RTX 5090 the fork plans exactly as
upstream.

The fork ids are identical, all three cases, to the `2026-09-25`, `2026-09-27` and `2026-09-28`
records (`md5sum` of the concatenated `.ids` files `6a25246257f0efef92393659cca2a288`): the same
token flipped the other way when upstream started quantizing Q in `d44ab584`.
