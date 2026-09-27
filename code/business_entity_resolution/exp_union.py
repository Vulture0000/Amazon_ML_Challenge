"""Greedy union: how much recall does each key family add per candidate pair?"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
B = importlib.import_module("02_blocking")
from common import WORK, Right, Source

split = sys.argv[1] if len(sys.argv) > 1 else "train"
NSEL = 40000
L = Source(f"{split}_s1")
R = Right(split)
tl = np.load(os.path.join(WORK, "gt_s1.npy"))
tr = np.load(os.path.join(WORK, "gt_right.npy"))
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(L), NSEL, replace=False))
mask = np.zeros(len(L), bool)
mask[sel] = True
pos = np.full(len(L), -1, np.int64)
pos[sel] = np.arange(len(sel))
gk = pos[tl] >= 0
TRUE = np.unique(pos[tl[gk]].astype(np.int64) << 24 | tr[gk])
N_TRUE = len(TRUE)
print(f"left={NSEL}  true pairs={N_TRUE}")
print("df quantiles for right-side keys:")
for nm, rk, roff in (("ntok", np.asarray(R.ntok), R.ntok_off),
                     ("atok", np.asarray(R.atok), R.atok_off),
                     ("npre", np.asarray(R.npre), R.npre_off)):
    kv, kc = B.key_counts(rk)
    tot = int(kc.sum())
    cs = np.cumsum(kc)
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        r = min(q, 0.9999999)
        print(f"  {nm:<6s} df@quantile {q:<6} = {cs[int(r*len(cs))]:>10,d}   n_keys={len(kv):,}")
    for cap in (100, 400, 2000, 10000, 50000):
        c = int(cs[min(cap, len(cs)) - 1])
        print(f"  {nm:<6s} cap={cap:<6d} keys_kept={min(cap,len(cs)):>9,d}  "
              f"postings={c:>12,d} ({c/tot:6.2%})")
    del kv, kc

fams = {}
lo, ln = B.filter_csr(L.ntok_off, np.asarray(L.ntok), mask)
ao, la = B.filter_csr(L.atok_off, np.asarray(L.atok), mask)
po, lp = B.filter_csr(L.npre_off, np.asarray(L.npre), mask)
fams["name_set"] = (np.arange(NSEL + 1, dtype=np.int32), B.set_key(ln, lo, 0x5DEECE66D).astype(np.int32),
                    np.arange(R.n + 1, dtype=np.int32), B.set_key(np.asarray(R.ntok), R.ntok_off, 0x5DEECE66D).astype(np.int32))
fams["addr_set"] = (np.arange(NSEL + 1, dtype=np.int32), B.set_key(la, ao, 0x2545F491).astype(np.int32),
                    np.arange(R.n + 1, dtype=np.int32), B.set_key(np.asarray(R.atok), R.atok_off, 0x2545F491).astype(np.int32))
fams["name_pre"] = (po, lp.astype(np.int32), R.npre_off, np.asarray(R.npre).astype(np.int32))
fams["name_tok"] = (lo, ln.astype(np.int32), R.ntok_off, np.asarray(R.ntok).astype(np.int32))
fams["addr_tok"] = (ao, la.astype(np.int32), R.atok_off, np.asarray(R.atok).astype(np.int32))

def run(name, loff, lk, roff, rk, cap):
    inv_k, inv_r = B.build_index(np.asarray(rk), roff, R.n)
    kv, kc = B.key_counts(np.asarray(rk))
    res = B.expand(loff, lk, inv_k, inv_r, kv, kc, cap, 1, max_out=300_000_000)
    del inv_k, inv_r, kv, kc
    if res is None:
        print(f"  {name:<20s} cap={cap:<6d}  (empty)")
        return np.zeros(0, np.int64)
    c = np.unique(res[0] << 24 | res[1])
    r = len(np.intersect1d(c, TRUE, assume_unique=True)) / N_TRUE
    print(f"  {name:<20s} cap={cap:<6d} pairs={len(c):>13,d} ({len(c)/NSEL:7.1f}/left) recall={r:.4f}")
    return c

RIDX = np.arange(R.n + 1, dtype=np.int32)
r_ntok, r_atok, r_npre = np.asarray(R.ntok), np.asarray(R.atok), np.asarray(R.npre)
l_hn = np.asarray(L.hnum)[mask]
r_hn = np.asarray(R.hnum)
LN = np.arange(NSEL + 1, dtype=np.int32)
hnum_f = (LN, B.num_key(l_hn, 0x1234567, l_hn > 0).astype(np.int32),
          RIDX, B.num_key(r_hn, 0x1234567, r_hn > 0).astype(np.int32))
fams["name_set"] = (LN, B.set_key(ln, lo, 0x5DEECE66D).astype(np.int32), RIDX,
                    B.set_key(r_ntok, R.ntok_off, 0x5DEECE66D).astype(np.int32))
fams["addr_set"] = (LN, B.set_key(la, ao, 0x2545F491).astype(np.int32), RIDX,
                    B.set_key(r_atok, R.atok_off, 0x2545F491).astype(np.int32))
fams["house_number"] = hnum_f

COMBOS = [
    [("addr_tok", 1000)],
    [("addr_tok", 1000), ("name_set", 2000)],
    [("addr_tok", 1000), ("name_set", 2000), ("addr_set", 100)],
    [("addr_tok", 1000), ("name_set", 2000), ("addr_set", 100), ("house_number", 2000)],
    [("addr_tok", 1000), ("name_set", 2000), ("addr_set", 100), ("house_number", 2000),
     ("name_pre", 2000)],
    [("addr_tok", 2000), ("name_set", 2000), ("addr_set", 100), ("house_number", 2000),
     ("name_pre", 2000)],
    [("addr_tok", 1000), ("name_tok", 2000), ("name_set", 2000), ("addr_set", 100),
     ("house_number", 2000)],
]
store = {}
print("\nindividual:")
for nm, cap in [("addr_tok", 1000), ("addr_tok", 2000), ("name_tok", 2000),
                ("name_pre", 2000), ("name_pre", 10000), ("house_number", 2000),
                ("name_set", 2000), ("addr_set", 100)]:
    store[(nm, cap)] = run(nm, *fams[nm], cap)
print("\ncombinations:")
for c in COMBOS:
    u = store[c[0]]
    for k in c[1:]:
        u = np.union1d(u, store[k])
    r = len(np.intersect1d(u, TRUE, assume_unique=True)) / N_TRUE
    print(f"  {str([k[0] for k in c]):<95s} pairs={len(u):>13,d} "
          f"({len(u)/NSEL:7.1f}/left) RECALL={r:.4f}")
