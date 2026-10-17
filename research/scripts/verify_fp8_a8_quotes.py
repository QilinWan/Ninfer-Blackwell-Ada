import subprocess
WAVE = 170
def r256(x): return (x + 255) // 256 * 256
def S(bt, br, split=0): return (bt, br, split)
# (band_top, schedule) ; schedule = (block_tokens, block_rows, split-K capable)
LADDERS = {
 (14336,5120):  (17, [(32,S(32,32)),(96,S(32,128)),(128,S(64,128)),(192,S(64,256)),(288,S(96,256)),(10**9,S(128,256,1))]),
 (16384,5120):  (17, [(32,S(32,32)),(64,S(64,128)),(128,S(64,128)),(192,S(192,128)),(384,S(128,256,1)),(512,S(128,128,1)),(10**9,S(128,256,1))]),
 (34816,5120):  (5,  [(64,S(64,128)),(128,S(64,128)),(192,S(64,256)),(10**9,S(128,256,1))]),
 (5120,6144):   (17, [(64,S(32,64)),(128,S(64,64)),(192,S(64,128)),(768,S(128,128,1)),(10**9,S(128,256,1))]),
 (5120,17408):  (17, [(64,S(32,64)),(128,S(64,128,1)),(256,S(128,128,1)),(384,S(192,128,1)),(512,S(128,256,1)),(768,S(192,128,1)),(10**9,S(128,256,1))]),
 (248320,5120): (10**9, []), (124160,5120): (10**9, []),
 (7168,5120):   (17, [(32,S(32,32)),(96,S(32,128)),(128,S(64,128)),(192,S(64,256)),(288,S(96,256)),(10**9,S(128,256,1))]),
 (8192,5120):   (17, [(32,S(32,32)),(64,S(64,128)),(128,S(64,128)),(192,S(192,128)),(384,S(128,256,1)),(512,S(128,128,1)),(10**9,S(128,256,1))]),
 (17408,5120):  (5,  [(64,S(64,128)),(128,S(64,128)),(192,S(64,256)),(10**9,S(128,256,1))]),
 (5120,3072):   (17, [(64,S(32,64)),(128,S(64,64)),(192,S(64,128)),(768,S(128,128,1)),(10**9,S(128,256,1))]),
 (5120,8704):   (17, [(64,S(32,64)),(128,S(64,128,1)),(256,S(128,128,1)),(384,S(192,128,1)),(512,S(128,256,1)),(768,S(192,128,1)),(10**9,S(128,256,1))]),
}
def requirement(N, K, T, lad):
    "bytes of split-K partials the A8 route needs for a call at T"
    for bound, (bt, br, split) in lad:
        if T <= bound:
            if not split: return 0
            tiles = (N // br) * ((T + bt - 1) // bt)
            tail = tiles % WAVE
            return WAVE * bt * br * 4 if (tail and tail <= WAVE // 2) else 0
    return 0
def peak(T, K, parts):
    return T * K + (r256(T * 4) if parts else T * 4) + parts

quotes = {}
for line in open('/tmp/quotes_fp8.txt'):
    n, k, t, q = map(int, line.split()); quotes[(n, k, t)] = q

under = []; over = []
for (N, K), (mn, lad) in LADDERS.items():
    for T in range(1, 4097):
        q = quotes.get((N, K, T))
        if q is None: continue
        p = requirement(N, K, T, lad) if T >= mn else 0
        m = peak(T, K, p) if T >= mn else 0
        if q < m: under.append((N, K, T, q, m))
        elif q > m: over.append((N, K, T, q - m))

def ranges(ts):
    out = []; a = b = ts[0]
    for t in ts[1:]:
        if t == b + 1: b = t
        else: out.append((a, b)); a = b = t
    out.append((a, b)); return out

print("under-quotes (danger):", len(under))
u = {}
for N, K, T, q, m in under: u.setdefault((N, K), []).append(T)
for kk in sorted(u): print("   ", kk, "n=", len(u[kk]), "ranges", ranges(u[kk])[:10])
print("over-quotes (reservations above the exact need):", len(over))
o = {}
for N, K, T, d in over: o.setdefault((N, K), {}).setdefault(d, []).append(T)
for kk in sorted(o):
    for d, ts in sorted(o[kk].items(), key=lambda x: -len(x[1]))[:3]:
        print("   ", kk, "delta", d, "n", len(ts), "T", ranges(ts)[:4])
