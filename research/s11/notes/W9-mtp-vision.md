# W9 MTP/DFlash2 + 图像（双卡）

- serve `--spec mtp --draft-tokens 3` + 768² 红图：**HTTP 200**，answer `" red"`，
  `mtp accepted 3/3 (100.0%)`。rank0 的 composed embedding（shifted ids + shifted 视觉行）走
  `mtp_prefill_chunk_tp2`，rank1 仍只吃 hidden。
  复测（两卡清净、本地 Qwen 栈已由用户关闭）：`--max-context 8192` 的 serve 同样 **HTTP 200**
  `"Red"`、`mtp accepted 3/3`，rank0 10.2 GiB / rank1 9.6 GiB。此前 8K 启动失败确系当时
  另一服务重载造成的显存争抢（预留 1.23 GB vs 可用 324 MB），不是本改动。
- `engine_vision_tp2_real mtp`（Engine 路径，8K 上下文）：rc=0，MTP 文本 + 二轮前缀复用
  （reused 30）+ 图像答案 "red" 全过。日志 `logs/vision-mtp.txt`。
- `engine_mtp_tp2_real`：Passed 88.59 s；`engine_mtp_tp2_optimized_real`：Passed 86.56 s
  （ctest，见 W11）。
- DFlash2：本制品（text+vision+mtp，无 dflash2）→ `missing component dflash2`；
  `engine_dflash2_tp2_real` 与 `vision_tp2_real dflash2` 现在按 **77 skip** 记录（不是假绿）。
- 未做：图像 prompt 的二轮前缀复用在“图像 token 正好在 frontier”时的 MTP 桥（tp2 仍要求
  一轴文本 RoPE，见 s11-notes 限制）。
