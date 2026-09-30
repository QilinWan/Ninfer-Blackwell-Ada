# 2026-09-30 — identical

| | |
|---|---|
| upstream | `Neroued/ninfer` at `d44ab584`, runner built with `BUILD_TESTING=OFF` (nvcc 13.1) |
| fork | branch `sync/upstream-d44ab584` = `f19e6821` (merge of upstream `d44ab584` into v0.2.3) |
| artifact | the same synthetic two-layer model as `2026-09-27`, 3,300,141,824 bytes, md5 `235749fafca92e11b760f1bfc7b8171a` (checked before the run) |
| hardware | one RTX 5070 Ti of the pair (device 0), CUDA 13.1 (nvcc V13.1.115), driver 595.91.07 |

`diff -r` between the two directories is empty: 128 / 128 / 118 generated ids (case 3 stops on an
end token), `md5sum` of the concatenated `.ids` files `54c26c125a64c657a7d738141b4a3c28` on both
sides. Against the `2026-09-28` record, cases 1 and 2 are unchanged; case 3 (INT8 KV, 6,144 prompt
tokens) diverges at generated token 100 and stops at 118 instead of 124. Upstream moved there
between `e31bc99b` and `d44ab584` (the INT8 causal attention now quantizes Q), and the fork follows
it exactly.
