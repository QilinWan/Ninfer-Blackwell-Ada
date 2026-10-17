#!/usr/bin/env python3
"""Cross-check accepted_lengths against the histogram and the counters in a request-log JSONL.

Contract (docs/serving.md 1199-1213): accepted_lengths is one hex char per decode step ('-' = a
step with no draft), histogram[i] = steps that accepted exactly i tokens, per_position[i] =
steps that accepted at least i+1. Identities checked when not truncated:
  sum(hist) == rounds; sum(i*hist[i]) == accepted_tokens;
  count(c in string) == hist[int(c,16)]; count('-') == fallback_steps.
Usage: python3 check_accepted_lengths.py req.jsonl [req2.jsonl ...]
"""
import json, sys, collections

def check(path):
    recs = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    done = [r for r in recs if r.get('event') == 'request_done'
            or r.get('type') == 'request_done' or 'speculative' in r]
    print(f'== {path}: {len(done)} request_done records')
    bad = 0
    for r in done:
        sp = r.get('speculative') or {}
        if not sp:
            continue
        al = sp.get('accepted_lengths')
        hist = (sp.get('accepted_length_histogram')
                or sp.get('accepted_length_lengths_histogram'))
        pos = sp.get('accepted_per_position') or []
        rounds = sp.get('rounds')
        acc = sp.get('accepted_tokens')
        fb = sp.get('fallback_steps')
        trunc = sp.get('accepted_lengths_truncated')
        rid = (r.get('request') or {}).get('request_id', r.get('request_id'))
        errs = []
        if hist is not None and rounds is not None and sum(hist) != rounds:
            errs.append(f'sum(hist)={sum(hist)} != rounds={rounds}')
        if hist is not None and acc is not None and sum(i * h for i, h in enumerate(hist)) != acc:
            errs.append(f'sum(i*hist)={sum(i*h for i,h in enumerate(hist))} != accepted={acc}')
        # per_position[i] counts the steps that accepted at least i+1 tokens (a suffix sum of
        # the histogram), so hist[0] = rounds - pos[0]; hist[i] = pos[i-1] - pos[i]; tail = pos[-1].
        if hist is not None and pos and rounds is not None:
            derived = [rounds - pos[0]]
            for i in range(1, len(pos)):
                derived.append(pos[i - 1] - pos[i])
            derived.append(pos[-1])
            if derived != list(hist):
                errs.append(f'hist={hist} != suffix-sum({pos}) -> {derived}')
        if al is None:
            state = 'null'
        else:
            state = 'present'
            if not trunc and hist is not None:
                cnt = collections.Counter(c for c in al)
                for c, n in cnt.items():
                    if c == '-':
                        if fb is not None and n != fb:
                            errs.append(f"'-' count {n} != fallback_steps {fb}")
                    else:
                        i = int(c, 16)
                        if i >= len(hist):
                            errs.append(f'char {c} out of hist range {len(hist)}')
                        elif n != hist[i]:
                            errs.append(f"count('{c}')={n} != hist[{i}]={hist[i]}")
                if rounds is not None and len(al) != rounds:
                    errs.append(f'len(string)={len(al)} != rounds={rounds}')
        if errs:
            bad += 1
        print(f'  req {rid}: {"OK" if not errs else "MISMATCH"} len={state} '
              f'rounds={rounds} acc={acc} fb={fb} hist={hist} '
              f'draft_window={sp.get("draft_window")} backend={sp.get("backend")}'
              + ('' if not errs else ' :: ' + '; '.join(errs[:3])))
    print(f'== {path}: {bad} mismatching records')
    return bad

if __name__ == '__main__':
    import sys
    sys.exit(1 if any(check(p) for p in sys.argv[1:]) else 0)
