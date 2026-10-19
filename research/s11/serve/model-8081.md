# :8081 — NInfer NVFP4 + DFlash2 behind the llama.cpp web UI

`model.sh` slot (the only change outside the engine): engine binary + libs under
`/tank/Ninfer/bin`, artifact under `/tank/models/kybrcore/...`, UI proxy under
`/home/ai-agent/ninfer-ui-proxy.py`.

```
define_model "Qwen3.8-27B" \
  "/tank/models/kybrcore/Qwen3.8-27B-QUASAR-NVFP4-NInfer" \
  "qwen3_8_27b_quasar.ninfer" \
  "" \
  "8081" "262144" \
  "--tp 2 --devices 0,1 --vision --vision-device 0 --kv-dtype nvfp4 \
   --max-context 262144 --kv-capacity 262144 --max-concurrency 4 \
   --spec dflash2 --draft-tokens 7 --lm-head-draft \
   --embedding-host --device-state-slots 2 \
   --default-max-tokens 32768 --default-thinking-budget 8192"
```

Process pair:

```bash
# engine (internal port; no listener on the LAN)
LD_LIBRARY_PATH=/tank/Ninfer/bin/lib /tank/Ninfer/bin/apps/ninfer-serve \
  /tank/models/kybrcore/Qwen3.8-27B-QUASAR-NVFP4-NInfer/qwen3_8_27b_quasar.ninfer \
  --host 127.0.0.1 --port 18091 --model-id Qwen3.8-27B \
  <the --tp 2 ... flags above>

# llama.cpp-shaped front end on the user port
/usr/bin/python3 /home/ai-agent/ninfer-ui-proxy.py \
  --host 0.0.0.0 --port 8081 --upstream http://127.0.0.1:18091 \
  --dist /tank/Strata/web-llama --model Qwen3.8-27B --ctx 262144 --slots 4 \
  --artifact /tank/models/kybrcore/Qwen3.8-27B-QUASAR-NVFP4-NInfer/qwen3_8_27b_quasar.ninfer \
  --template-file /home/ai-agent/chat_template_qwen38-27b.jinja
```

The proxy serves `index.html`/`_app`, answers `/props` (`n_ctx`, modalities, template),
`/slots`, `/models/*` and forwards `/v1/chat/completions` (SSE), `/v1/responses` and
`/v1/messages`; generation requests without `model` get the served alias injected.

Measured (2x RTX 5060 Ti, 256K, DFlash2 K=7, 4 lanes): 240 tok/s on predictable text,
78 tok/s on open-ended prose, 125 tok/s on a real image+multi-turn request; 4 concurrent
512-token streams = 627 tok/s aggregate; VRAM 15.4 / 11.6 GiB. 512K does not fit
(needs 6.85 GB vs 4.70 GB available after weights).
