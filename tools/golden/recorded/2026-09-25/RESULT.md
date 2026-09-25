# 2026-09-25 — identical

| | |
|---|---|
| upstream | `Neroued/ninfer` at `bace20dc` (the fork's base), detached worktree, Release, `sm_120a` |
| fork | engine at `main` = `2e7f7d3a` (runner and `write` mode from `d670eb01`, tools only) |
| artifact | synthetic two-layer model written by `ninfer_qwen3_5_text_context_tp2_test write`, 3,300,141,824 bytes |
| hardware | one RTX 5070 Ti of the pair (device 0), CUDA 13.1.1, driver 595.91.07 |

`diff -r` between the two directories is empty: 128 / 128 / 124 generated ids (case 3 stops on an
end token), `md5sum` of the concatenated `.ids` files `1977ff362092123d5ec0e0aebccbdaf7` on both sides.
