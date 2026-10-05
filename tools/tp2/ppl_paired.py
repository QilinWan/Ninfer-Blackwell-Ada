#!/usr/bin/env python3
"""Paired per-window comparison of two ninfer-perplexity reports on the same corpus and window plan.

    tools/tp2/ppl_paired.py REFERENCE.json CANDIDATE.json [--alpha 0.05] [--max-delta-pct 0.5]

Every window of a `ninfer-perplexity --output` report is a fresh prefill that scores its own
disjoint tokens, so the per-window differences of total NLL are (about) independent samples of the
perturbation. The mean difference per scored token is d = sum(dNLL) / N, the perplexity ratio is
exp(d), and its standard error comes from the spread between windows: SE = sqrt(n) * stdev(dNLL) / N
over n windows. The two-sided p of the paired t (Student, n - 1 degrees of freedom) and an exact
sign test are reported with the 95 % interval of the perplexity change.

The first word of the first line is the verdict the gate prints:
  PASS  the tables are bit-identical, or the candidate is not worse at the alpha level (it is
        better, or worse with p > alpha) and |change| < --max-delta-pct;
  WARN  otherwise (significantly worse, or a change at least --max-delta-pct in either direction);
  FAIL  the two reports do not describe the same windows.
A second line gives the change per domain. Standard library only (the gate's python3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from statistics import stdev


def _betacf(a: float, b: float, x: float) -> float:
    # Continued fraction of the regularized incomplete beta (modified Lentz).
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """P(|T| >= |t|) for Student's t with df degrees of freedom."""
    if df <= 0 or t != t:
        return float("nan")
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(q: float, df: int) -> float:
    """The q-quantile (q > 0.5) of Student's t, by bisection on the two-sided tail."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) > 2.0 * (1.0 - q):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def sign_test_p(pos: int, neg: int) -> float:
    m = pos + neg
    if m == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2.0 * sum(math.comb(m, j) for j in range(k + 1)) / 2.0 ** m)


def windows(report: dict) -> dict:
    out = {}
    for stream in report["streams"]:
        for w in stream["windows"]:
            out[(stream["id"], w["index"])] = (w["scored_tokens"], w["total_nll"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-delta-pct", type=float, default=0.5)
    args = ap.parse_args(argv)
    ref, new = (json.load(open(path)) for path in (args.reference, args.candidate))
    wr, wn = windows(ref), windows(new)
    keys = sorted(wr)
    if set(wr) != set(wn) or any(wr[k][0] != wn[k][0] for k in keys):
        print("FAIL the reports score different windows (corpus, context or stride differ)")
        return 1
    n = len(keys)
    scored = sum(wr[k][0] for k in keys)
    diff = [wn[k][1] - wr[k][1] for k in keys]
    p_ref, p_new = ref["overall"]["perplexity"], new["overall"]["perplexity"]
    pos, neg = sum(x > 0 for x in diff), sum(x < 0 for x in diff)
    same = n - pos - neg
    if same == n:
        print(f"PASS identical: {n} windows, perplexity {p_new:.6f}")
        return 0
    mean = sum(diff) / scored                     # nat per scored token
    change = 100.0 * math.expm1(mean)             # = 100 (PPL_new / PPL_ref - 1)
    se = math.sqrt(n) * stdev(diff) / scored if n > 1 else float("nan")
    if se == 0.0:
        t, p = math.copysign(math.inf, mean), 0.0
    else:
        t = mean / se
        p = t_two_sided_p(t, n - 1)
    half = t_quantile(0.975, n - 1) * se if n > 1 else float("nan")
    low, high = 100.0 * math.expm1(mean - half), 100.0 * math.expm1(mean + half)
    psign = sign_test_p(pos, neg)
    worse = mean > 0.0 and p <= args.alpha
    verdict = "WARN" if worse or abs(change) >= args.max_delta_pct else "PASS"
    reason = ("significantly worse" if worse else
              f"|change| >= {args.max_delta_pct} %" if verdict == "WARN" else
              "not worse at alpha" if mean > 0.0 else "better")
    print(f"{verdict} paired ({reason}): perplexity {p_ref:.6f} -> {p_new:.6f} ({change:+.3f} %), "
          f"95 % CI [{low:+.3f}, {high:+.3f}] %, t {t:+.2f} p {p:.3f} (n {n}), "
          f"signs +{pos}/-{neg}/={same} p {psign:.3f}")
    dr = {d["domain"]: d["perplexity"] for d in ref.get("domains", [])}
    dn = {d["domain"]: d["perplexity"] for d in new.get("domains", [])}
    print("  " + " · ".join(f"{k} {dr[k]:.4f}->{dn[k]:.4f} ({100 * (dn[k] / dr[k] - 1):+.3f} %)"
                            for k in sorted(dr) if k in dn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
