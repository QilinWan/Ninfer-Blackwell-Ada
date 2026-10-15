# 2026-09-28 — identical

| | |
|---|---|
| upstream | `Neroued/ninfer` at `e31bc99b`, the same clone and runner build as `2026-09-27` |
| fork | `main` = `44a58463` (v0.2.1 plus the pipelined mailbox exchange kernel), the v0.2.2 candidate build |
| artifact | the same synthetic two-layer model as `2026-09-27`, 3,300,141,824 bytes, md5 `235749fafca92e11b760f1bfc7b8171a` (checked before the run) |
| hardware | one RTX 5070 Ti of the pair (device 0), CUDA 13.1.1 (nvcc V13.1.115), driver 595.91.07 |

`diff -r` between the two directories is empty: 128 / 128 / 124 generated ids (case 3 stops on an
end token), `md5sum` of the concatenated `.ids` files `6a25246257f0efef92393659cca2a288` on both
sides. The ids are also identical to the `2026-09-27` record: nothing merged since `b3f93dd6`
(v0.2.0, v0.2.1 and the pipelined mailbox kernel) changed the tp 1 path.
