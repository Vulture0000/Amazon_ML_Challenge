"""Measure the recall / candidate-volume trade-off of each blocking family."""
import sys
import os

import numpy as np

import common
from common import WORK, Right, Source
import importlib
B = importlib.import_module("02_blocking")

split = sys.argv[1] if len(sys.argv) > 1 else "train"
L = Source(f"{split}_s1")
R = Right(split)
tl = np.load(os.path.join(WORK, "gt_s1.npy"))
tr = np.load(os.path.join(WORK, "gt_right.npy"))
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(L), 40000, replace=False))
mask = np.zeros(len(L), bool)
mask[sel] = True
pos = np.full(len(L), -1, np.int64)
pos[sel] = np.arange(len(sel))
gk = pos[tl] >= 0
TL, TR = pos[tl[gk]], tr[gk]
TRUE = np.unique(TL.astype(np.int64) << 24 | TR)
N_TRUE = len(TRUE)
print(f"left sample={len(sel)}  true pairs in sample={N_TRUE}")

l_ntok_off, l_ntok = B.filter_csr(L.ntok_off, np.asarray(L.ntok), mask)
l_atok_off, l_atok = B.filter_csr(L.atok_off, np.asarray(L.atok), mask)
r_ntok, r_atok = np.asarray(R.ntok), np.asarray(R.atok)

fams = {
    "name_token": (l_ntok_off, l_ntok, R.ntok_off, r_ntok),
    "addr_token": (l_atok_off, l_atok, R.atok_off, r_atok),
    "house_number": (np.arange(len(sel) + 1, dtype=np.int32),
                     B.num_key(np.asarray(L.hnum)[mask], 0x1234567,
                               np.asarray(L.hnum)[mask] > 0).astype(np.int32),
                     np.arange(R.n + 1, dtype=np.int32),
                     B.num_key(np.asarray(R.hnum), 0x1234567,
                               np.asarray(R.hnum) > 0).astype(np.int32)),
    "name_set": (np.arange(len(sel) + 1, dtype=np.int32),
                 B.set_key(l_ntok, l_ntok_off, 0x5DEECE66D).astype(np.int32),
                 np.arange(R.n + 1, dtype=np.int32),
                 B.set_key(r_ntok, R.ntok_off, 0x5DEECE66D).astype(np.int32)),
    "addr_set": (np.arange(len(sel) + 1, dtype=np.int32),
                 B.set_key(l_atok, l_atok_off, 0x2545F491).astype(np.int32),
                 np.arange(R.n + 1, dtype=np.int32),
                 B.set_key(r_atok, R.atok_off, 0x2545F491).astype(np.int32)),
}
for cap_list in ([12, 40, 100, 400, 2000, 10000],):
    for name, (loff, lk, roff, rk) in fams.items():
        inv_k, inv_r = B.build_index(np.asarray(rk), roff, R.n)
        kv, kc = B.key_counts(np.asarray(rk))
        for cap in cap_list:
            res = B.expand(loff, lk, inv_k, inv_r, kv, kc, cap, 1, max_out=200_000_000)
            if res is None:
                print(f"  {name:<12s} cap={cap:<6d} 0 pairs")
                continue
            Lp, Rp = res
            cand = np.unique(Lp << 24 | Rp)
            hit = np.intersect1d(cand, TRUE, assume_unique=True)
            print(f"  {name:<12s} cap={cap:<6d} pairs={len(cand):>12,d} "
                  f"({len(cand)/len(sel):8.1f}/left)  recall={len(hit)/N_TRUE:.4f}")
        del inv_k, inv_r, kv, kc
