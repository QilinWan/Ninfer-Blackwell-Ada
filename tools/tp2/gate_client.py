#!/usr/bin/env python3
"""Client of tools/tp2/gate.sh for the serve stages.

  gate_client.py run HOST:PORT PROMPTS.jsonl LIMIT OUT.jsonl
      sends the first LIMIT prompts (0 = all) one at a time, T=0, 128 tokens, thinking off, and
      writes one {"id", "http", "text", "md5", "completion_tokens", "finish"} record per prompt
  gate_client.py summarize OUT.jsonl REQUEST_LOG.jsonl
      prints the stage's numbers (texts, decode tok/s, acceptance, median ms/round)
  gate_client.py compare REF.jsonl REF_LOG.jsonl OUT.jsonl OUT_LOG.jsonl
      prints "PASS|WARN|FAIL detail": FAIL when a text differs, WARN when the median ms/round moves
      by more than 1 % against the reference, PASS otherwise
"""
import hashlib
import json
import statistics
import sys
import urllib.request


def run(host, prompts_path, limit, out_path):
    prompts = [json.loads(line) for line in open(prompts_path, encoding="utf-8")]
    if int(limit) > 0:
        prompts = prompts[: int(limit)]
    errors = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in prompts:
            body = {"model": "gate", "messages": [{"role": "user", "content": p["text"]}],
                    "max_tokens": 128, "temperature": 0, "reasoning_effort": "none"}
            req = urllib.request.Request("http://%s/v1/chat/completions" % host, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "X-Ninfer-Client": "gate/tp2"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    j = json.loads(r.read())
                    code = r.status
                c = j["choices"][0]["message"].get("content") or ""
                rec = {"id": p["id"], "http": code, "text": c, "md5": hashlib.md5(c.encode()).hexdigest()[:8],
                       "completion_tokens": j["usage"]["completion_tokens"], "finish": j["choices"][0]["finish_reason"]}
            except Exception as e:  # noqa: BLE001 - recorded, the comparison fails on it
                errors += 1
                rec = {"id": p["id"], "http": getattr(e, "code", 0), "error": str(e)[:200]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    return 1 if errors else 0


def stats(log_path):
    done = []
    for line in open(log_path, encoding="utf-8"):
        o = json.loads(line)
        if o.get("event") == "request_done":
            done.append(o)
    tokens = sum(max(o["result"]["completion_tokens"] - 1, 0) for o in done)
    decode = sum(o["timings_seconds"]["decode"] for o in done)
    spec = [o.get("speculative") or {} for o in done]
    accepted = sum(s.get("accepted_tokens", 0) for s in spec)
    drafted = sum(s.get("drafted_tokens", 0) for s in spec)
    per_round = [1000.0 * o["timings_seconds"]["decode"] / s["rounds"]
                 for o, s in zip(done, spec) if s.get("rounds", 0) >= 8]
    return {"n": len(done), "tps": tokens / decode if decode else 0.0,
            "acc": 100.0 * accepted / drafted if drafted else 0.0,
            "ms_round": statistics.median(per_round) if per_round else 0.0}


def texts(path):
    return {o["id"]: o for o in (json.loads(line) for line in open(path, encoding="utf-8"))}


def describe(s):
    return "decode %.2f t/s, acceptance %.2f %%, ms/round median %.3f (n=%d)" % (s["tps"], s["acc"], s["ms_round"], s["n"])


def summarize(out_path, log_path):
    t = texts(out_path)
    ok = sum(1 for o in t.values() if o.get("http") == 200)
    print("%d/%d answered; %s" % (ok, len(t), describe(stats(log_path))))
    return 0


def compare(ref_path, ref_log, out_path, out_log):
    ref, new = texts(ref_path), texts(out_path)
    same = [k for k in ref if k in new and new[k].get("text") == ref[k].get("text") and new[k].get("http") == 200]
    diff = [k for k in ref if k not in same]
    rs, ns = stats(ref_log), stats(out_log)
    delta = 100.0 * (ns["ms_round"] / rs["ms_round"] - 1.0) if rs["ms_round"] else 0.0
    status = "FAIL" if diff else ("WARN" if abs(delta) > 1.0 else "PASS")
    detail = "texts %d/%d identical%s; %s; ms/round %+.2f %% vs ref %.3f, acceptance ref %.2f %%" % (
        len(same), len(ref), (" (differ: %s)" % " ".join(diff[:8])) if diff else "", describe(ns), delta,
        rs["ms_round"], rs["acc"])
    print(status, detail)
    return 0


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    sys.exit({"run": run, "summarize": summarize, "compare": compare}[cmd](*args))
