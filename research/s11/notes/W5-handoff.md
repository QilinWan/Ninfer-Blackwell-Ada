# W5 机制验证：图像请求 200（双卡）

改造后二进制 `/data/s11/build-120a-cu13`，serve 参数同 W1（8K/32K 上下文均可）。

- 768² 红图 → **HTTP 200**，answer `"Red"`，prompt 601 tokens，completion 2。
- 768² 蓝图 → **HTTP 200**，answer `" blue"`。
- 同图贪心两跑：`generated_token_ids` 完全一致（`[5844, 248046]`），第二次 `cached_tokens=592`
  （serve 默认跨请求前缀复用生效）。
- 文本请求在图像之后仍正常（文本与 Vision 引擎的 32 token 输出逐字一致，见测试输出）。

机制：rank0 `prepare_chunk` 编码 → `cudaMemcpy2DAsync`(UVA, 无 P2P) 到 rank1 `ws[1]` 的
`[H, visual_rows]` → 两卡各自 `ops::scatter`；3 轴 MRoPE 两卡同源。
日志：`logs/serve-new.log`、`logs/req-new.jsonl`；结论：不再 500，答案内容正确。
