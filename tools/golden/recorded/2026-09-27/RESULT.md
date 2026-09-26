# 2026-09-27 — identical

| | |
|---|---|
| upstream | `Neroued/ninfer` at `e31bc99b` (the `master` merged into the fork), clone with the runner added to `apps/`, Release, `sm_120a` |
| fork | branch `up-merge` = `b3f93dd6` (merge of `e31bc99b` into `370a6670` plus the tp 2 fixes) |
| artifact | synthetic two-layer model written by `ninfer_qwen3_5_text_context_tp2_test write` of the fork build, 3,300,141,824 bytes, md5 `235749fafca92e11b760f1bfc7b8171a` |
| hardware | one RTX 5070 Ti of the pair (device 0), CUDA 13.1.1 (nvcc V13.1.115), driver 595.91.07 |

`diff -r` between the two directories is empty: 128 / 128 / 124 generated ids (case 3 stops on an
end token), `md5sum` of the concatenated `.ids` files `6a25246257f0efef92393659cca2a288` on both sides.

Against the `2026-09-25` record (both sides at `bace20dc`), the ids of cases 1 and 2 now diverge
after 45 and 52 identical tokens and case 3 is unchanged. The change comes from upstream, since
the fork still matches it. Candidates, not isolated: from `0784e76f` every prefill of 16 or more
tokens takes the two-stage chunked Gated DeltaNet (before, only whole 64-token chunks did), and
the FP8 and BF16 linear routes were re-expressed on the unified templates.
