# DFlash2 + QUASAR W4A4（kybrcore 制品，2×5060Ti）

制品：`/tank/models/kybrcore/Qwen3.8-27B-QUASAR-NVFP4-NInfer/qwen3_8_27b_quasar.ninfer`
19,782,448,132 B，sha256 `d196bf1d…63594`，v3，components text+vision+mtp+dflash2。
- W4A4 已确认：`inspect --bindings` 里 512 个投影 `activation_policy=AllowA4`；
  KV 用 `--kv-dtype nvfp4`（引擎 A4 走 `mma.sync.kind::mxf4nvf4`，T≥4）。
- 启动：`--tp 2 --devices 0,1 --vision --vision-device 0 --kv-dtype nvfp4
  --max-context 262144 --kv-capacity 262144 --max-concurrency 4
  --spec dflash2 --draft-tokens 7 --lm-head-draft --embedding-host --device-state-slots 2`
  显存 15.4 / 11.6 GiB（`--embedding-host` + 2 额外槽位是 256K 能装下的关键）。

## 256K 实测
| 场景 | decode | DFlash2 接受率 |
|---|---|---|
| 可预测文本（数字序列），单流 | **239.9 tok/s** | 446/448 (99.6%) |
| 开放写作（Rome 短文），单流 | 78.0 tok/s | 311/1,378 (22.6%) |
| 图像识别 | 200，`" red"` | — |
| 4 并发 ×512 | **628 tok/s 聚合**，157/路，单路 ~175 tok/s | 99.6% |
| 6 并发 ×512（slots 0） | 511 tok/s 聚合（更差，已被队列摊薄） | ~99% |

结论：4 并发是甜点（比 6 并发聚合高 23%且保留 2 个检查点槽位）。

## 512K
**装不下**：512K + DFlash2 + Vision + c4 需 6.85 GB，可用 4.70 GB；
即使 c1/slots0/512 chunk 也需 5.94 GB；去掉 vision 仍需 6.06 GB。
384K/c4 需 5.64 GB、320K/c4 需 5.04 GB，都超；256K 是这件的实际上限。

## 修掉的一个真 bug（多模态 + DFlash2）
首次用 UI 的"6 条消息 + 2 张图 + tools + thinking"请求时引擎 500 并退出：
`text/layers/0 prefill tp2 columns=1024: std::bad_alloc`。
根因：tp 工作区计划把 rank1 的 `[hidden, chunk]` 接收块**排在 target_body 之后**，
而 layout 的峰值由层内 scratch 决定 ⇒ 该块从不计入峰值，多模态 DFlash prefill 的
运行时分配超容量。修复：把该块移到 `target_body` 之前（`startup.cpp`，提交 `52ea073b`）。
验证：同一请求现在 200（4,195 prompt / 59 output，decode 124.8 tok/s，accept 45.9%）。
复跑 `ctest -R tp2`：10 个中 8 passed / 2 skipped（DFlash2 制品缺组件的那两个），310 s。

## 最终 8081 配置（256K，DFlash2 K=7）
- 显存 15.4 / 11.6 GiB；单流 240 tok/s（可预测）/ 78 tok/s（开放写作）；
- 4 并发 627 tok/s 聚合、157 tok/s/路；UI（8081）图像与多图会话均 200。
