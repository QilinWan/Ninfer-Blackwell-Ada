# GSM8K 判分底座（S9）· 说明与权威口径

**状态**：✅ 判分链路已跑通并自检（CPU，零 GPU，不碰现网）。**分母数值（`--tp 1` 的 500 题分数）留到双卡窗口内产出** —— 27B 装不进单卡（任务书 §1 S9 的口径）。

## 文件

| 文件 | 作用 |
|---|---|
| `data/gsm8k_test.jsonl` | 官方 test，**1319 行**，SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| `data/gsm8k_train.jsonl` | 官方 train，**7473 行**，SHA256 `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`（k-shot 提示的示例来源） |
| `run_gsm8k.py` | 客户端：OpenAI 兼容 `/v1/chat/completions`，贪心、k=8 提示、stop=`["\n\nQ:"]`；落 `meta` 行 + 逐题 JSONL |
| `grade.py` | 判分核：`####` 取 gold；提取 `The answer is X`（取不到则全文最后一个数字）；**canonical 字符串**为主判据 + numeric（相对误差 1e-6）为副判据 |
| `selftest.py` | 判分链路自检（桩服务三种应答：gold / messy / perturb） |
| `selftest_*.jsonl` `selftest_*.report.json` | 自检证据（保留） |

## 权威口径（D3 判定用，写死，不得中途更改）

1. **子集**：`test.jsonl` 文件顺序**前 500 题**（`--count 500`）。每次跑都落 `subset_digest`
   = SHA256(`index|question|gold` 逐行串联)。**分母与分子的 digest 必须相同才可比**（`grade.py` 会回显）。
2. **解码**：`temperature=0`（贪心）、`max_tokens` 固定并记录（默认 512）、`stop=["\n\nQ:"]`。
   思考型制品需把 `--max-tokens` 放到不截断推理（在 `engine_tag` 里写清取值）。
3. **提示**：同一 `--shot`（默认 8，取自 train 前 8 题）；`--extra-body` 传引擎侧开关时必须同串写入 `engine_tag`。
4. **`--engine-tag` 是强制项**，分母/分子各写一条，格式：
   `"<git尖> <tp> <构建档案> <服务关键开关>"`，例：
   - 分母：`v-follow-81c8ce0-r0 + G-TP2-1 --tp 1 --kv auto --lanes 1`
   - 分子：同一构建尖 `--tp 2`（**唯一允许变化的就是 `--tp`**，其余逐项相同）
5. **判定**：TP2 相对 `--tp 1` 分母的 **canonical_acc 掉幅 < 1 个百分点**（任务书 §2 D3 口径；**ppl 不得单独作判据**）。
   主判据 = canonical；numeric 只用于发现格式退化（两口径差得远 ⇒ 输出格式漂了，要单独看）。
6. 报告必须同时给：`n`、`canonical_correct/acc`、`numeric_correct/acc`、`unmatched_gold`、
   `request_errors`、`subset_digest`、每题 `finish_reason`（`length` 占比高 ⇒ 截断污染判分，需加大 `max_tokens` 重跑）。

## 跑法

```bash
cd relay/gsm8k
# 自检（零负载，随时可跑）
python3 selftest.py            # 期望三行全 OK

# 窗口内取分母（--tp 1），再取分子（--tp 2），最后判分
python3 run_gsm8k.py --base-url http://127.0.0.1:<port> --model <name> \
  --count 500 --shot 8 --max-tokens 512 --concurrency 1 \
  --engine-tag "<尖> --tp 1 …" --out preds_tp1.jsonl
python3 run_gsm8k.py … --engine-tag "<同一尖> --tp 2 …" --out preds_tp2.jsonl
python3 grade.py --pred preds_tp1.jsonl --report report_tp1.json
python3 grade.py --pred preds_tp2.jsonl --report report_tp2.json
```

## 未验证项（不要当成已完成）

- 客户端对**真实** `ninfer-serve` 的兼容性未测（本会话不碰 GPU0 现网，只用本机桩服务证伪/证实判分链）：
  需要窗口内首跑核 `/v1/models` 的 `engine_meta` 字段、`reasoning_content` 的键名、以及 `stop` 生效行为。
- 500 题子集 = 文件顺序前 500（非随机）；若要换随机子集，必须换 digest 并**重取分母**。
- 并发 >1 时服务端批处理会影响逐题时延（不影响判分正确性）；性能数（t/s）不在本底座职责内。
