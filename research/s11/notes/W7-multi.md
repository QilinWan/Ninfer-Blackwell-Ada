# W7 多图 / 非方形 / 确定性（双卡）

`engine_vision_tp2_real` parity 模式新增并通过（`logs/vision-parity.txt`）：

- 单图红/蓝：`" red"` / `" blue"`，各 0.36 s。
- **一条消息两张图**（红+蓝，同一 prompt 两次编码 + 两次 handoff）→ `"red, blue"`，
  两跑 token ids 一致。
- **非方形 1024×512 纯绿图** → `" green"`（patch 网格/重采样走不同形状）。
- 文本不变性：Vision 引擎与无 Vision 引擎的文本 32 token ids 逐字一致。
- 逐图贪心重复：token 完全相同。

未做（诚实记录）：视频；`--max-vision-tokens` 真正改变预处理尺寸（本树该选项不参与预处理，
见 s11-notes 的既有缺陷）。
