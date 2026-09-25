#!/bin/bash
# Needle-in-haystack past the artifact native window, exercising YaRN extrapolation.
# Usage: needle-extrapolation.sh <endpoint> <model-id> [tokens] [port]
# Builds a haystack of ~<tokens> tokens with one unique fact in the middle and asks for it.
# Run the server with a YaRN factor large enough: max_context >= native * factor.
set -euo pipefail
EP=${1:-http://127.0.0.1:8080}; MODEL=${2:-EfficientThink-27B}; TOK=${3:-280000}
export PATH=/root/miniconda3/bin:$PATH
python3 - "$TOK" <<'PY'
import sys
n = int(sys.argv[1]) // 34          # 34 filler sentences ~= 1K tokens
filler = ("The archive room held countless ledgers. Clerks moved between the shelves, "
          "recording shipments of grain, wool and salt in patient columns. ")
needle = " IMPORTANT FACT: the vault access code for the northern depot is ZX-8147-QQ. "
parts = []
for i in range(n):
    parts.append(filler)
    if i == n // 2:
        parts.append(needle)
open("/tmp/haystack.txt", "w").write("".join(parts))
print("haystack chars:", len(open("/tmp/haystack.txt").read()))
PY
python3 - "$MODEL" <<'PY' > /tmp/needle_req.json
import json, sys
hay = open("/tmp/haystack.txt").read()
q = hay + "\n\nQuestion: What is the vault access code for the northern depot? Answer with the code only."
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": q}],
                  "max_tokens": 400, "temperature": 0}))
PY
curl -s -m 3600 "$EP/v1/chat/completions" -H "Content-Type: application/json" \
  --data-binary @/tmp/needle_req.json > /tmp/needle_resp.json
python3 - <<'PY'
import json
c = json.load(open("/tmp/needle_resp.json"))["choices"][0]["message"]
t = (c.get("content") or "") + " || " + (c.get("reasoning_content") or "")
print("HIT ZX-8147-QQ:", "ZX-8147-QQ" in t)
print("tail:", t[-200:].replace("\n", " "))
PY
