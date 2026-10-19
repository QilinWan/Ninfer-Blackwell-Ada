# W1 复现 500 + 纯文本基线（双卡，DeepSeek）

制品：`/tank/models/Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/qwen3_8_27b_quasar_nvfp4.ninfer`
（17,397,699,076 B，sha256 `808a0fa3…b570` 与计划书一致；用户另一会话下载，本会话只读）。

基线二进制：`/data/s11/build-base`（worktree `/data/s11/tree-base` @ b55da868），全部 996 目标 rc=0。

命令（`tools/serve-start.sh base /data/s11/build-base`）：
`ninfer-serve <artifact> --tp 2 --devices 0,1 --vision --vision-device 0 --kv-dtype nvfp4
 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --prefill-chunk 1024`
（`--max-vision-tokens 1024` 在本树对预处理不起作用，见 s11-notes 的既有缺陷）。

结果：
- 文本 `What is 17*23?` → HTTP 200（16 token 被 thinking 用尽；之后请求加 `enable_thinking=false`）。
- 图像（768² 纯 PPM）→ **HTTP 500**，`error.message` 逐字
  `multimodal prefill is not implemented at tensor-parallel width 2`；随后引擎整体失败、server 自停（预期）。
- 纯文本 60 题：decode 均值 **46.0 tok/s**、prefill **434.6 tok/s**（`logs/serve-base.log`），
  token ids 存 `logs/ids-base.json`（W3 对照）。
- 显存：rank0 12.5 GiB / rank1 9.5 GiB（8K 上下文时）。

下一步：W3 用同一 60 题在改造后二进制上逐字对照。
