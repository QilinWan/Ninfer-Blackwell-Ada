# W2/W4 代码 — tp2 多模态 handoff（方案 A）

改动（`b55da868` 之上，未提交）：
- `execution/text.cpp` `prefill_impl_tp2`：接受 `MultimodalPrefill*`；rank0 `prepare_chunk` 编码后
  以 **cudaMemcpy2DAsync（跨卡 UVA，无 P2P 也可）** 把本 chunk 的 `[H,视觉行]` 拷到 rank1 的
  `ws[1]` 暂存；两卡各自 `ops::scatter` 同一批行；3 轴 MRoPE 从 prompt 位置逐轴 host→device
  写到两卡（axis-major [3,T] 约定）；rank1 的 positions/rope 绑定沿用 peer_*。
- `prefill_chunk(...vision...)` 去掉 `require_single_device`，tp2 走新路径（含 DFlash sink 版本）。
- MTP：`mtp_forward_stem_tp2` 增 `input_embeddings`（仅 rank0 的 packed 半边用 composed embedding）；
  `mtp_prefill_chunk_tp2` 支持 [T,3] rope（末列 axis-major 提取）；`mtp_bridge_and_propose` 解除
  composed-embedding 拒绝（rank1 不 embed，无需 handoff；仍要求文字 RoPE）。
- `planning/startup.cpp` tp 工作区：text_prefill 加 3 轴 rope + scatter chunk + rank1 `[H,chunk]`
  接收缓冲；mtp_prefill 加 rank0 `[H,chunk]` composed embedding + shifted 视觉 scatter 索引。
  容量仍 `max(general, vision)`（vision handoff 放在 general 之上）。

编译：全量 rc=0；`ninfer_qwen3_5_text_context_tp2_test`（双卡、合成权重）prefill/decode
max|tp2-tp1| ≤ 0.1875（tolerance 0.25）→ 纯文本 tp2 数值未回归。

下一步（需制品）：W1 复现 500 基线 / W5 图像 200 / W3 文本零回归 / W7 多图确定性 / W9 MTP 图像。
