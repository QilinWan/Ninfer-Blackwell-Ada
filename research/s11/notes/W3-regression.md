# W3 纯文本 tp 2 零回归

同一制品、同一 serve 参数、同一 60 题（`tools/text-prompts-60.json`，greedy、seed 42、
`enable_thinking=false`、max 24），旧（b55da868）与新（本分支）各跑一遍，
按 `request_done.result.generated_token_ids` 逐字比对：

- **60/60 完全一致，0 差异**（`logs/ids-base.json` vs `logs/ids-new.json`）。
- 新二进制 decode 均值见 `logs/serve-new.log`（同一硬件窗口，未单独调参）。

为什么应逐字一致：tp2 文本 chunk 的分配序列与计算图未变（`visual_rows == 0` 时
`text_prefill_roots` 的 scatter=0、rope_axes 逻辑与旧码相同），新代码只在有视觉行时才多分配
`[H, 视觉行]` 接收缓冲并插一次跨卡拷贝。

结论：P1/P2 改造未改变纯文本 tp 2 的数值路径。
