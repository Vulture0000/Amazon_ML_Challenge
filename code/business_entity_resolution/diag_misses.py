"""Diagnose blocking misses: which key families would have caught them?"""
import os
import sys

import numpy as np

from common import WORK, Right, Source

tl = np.load(os.path.join(WORK, "gt_s1.npy"))
tr = np.load(os.path.join(WORK, "gt_right.npy"))
L = Source("train_s1")
R = Right("train")
ul = np.load(os.path.join(WORK, "cand_trainsm_l.npy"))
ur = np.load(os.path.join(WORK, "cand_trainsm_r.npy"))
lidx = np.load(os.path.join(WORK, "cand_trainsm_leftidx.npy"))
pos = np.full(len(L), -1, np.int64)
pos[lidx] = np.arange(len(lidx))

pred = np.unique(ul.astype(np.int64) << 24 | ur)
keep = pos[tl] >= 0
tl2, tr2 = pos[tl[keep]], tr[keep]
true = np.unique(tl2.astype(np.int64) << 24 | tr2)
miss = np.setdiff1d(true, pred, assume_unique=True)
print(f"missed {len(miss)} / {len(true)} = {len(miss)/len(true):.3f}\n")

rng = np.random.default_rng(0)
show = rng.choice(len(miss), 30, replace=False)
lrow = miss[show] >> 24
rrow = miss[show] & ((1 << 24) - 1)
n1 = L.names(lidx[lrow])
a1 = L.addrs(lidx[lrow])
r_in2 = rrow < R.n2
n2 = R.names(rrow)
a2 = R.addrs(rrow)

h1 = np.asarray(L.hnum)[lidx[lrow]]
h2 = np.asarray(R.hnum)[rrow]
z1 = np.asarray(L.zpc)[lidx[lrow]]
z2 = np.asarray(R.zpc)[rrow]
print("right source split: s2=%d s3=%d" % (r_in2.sum(), (~r_in2).sum()))
print("housenum equal: %.3f   both nonzero: %.3f" % (
    float((h1 == h2).mean()), float(((h1 > 0) & (h2 > 0)).mean())))
print("zip equal: %.3f   both nonzero: %.3f" % (
    float((z1 == z2).mean()), float(((z1 > 0) & (z2 > 0)).mean())))
for i in range(len(show)):
    t1, t2 = set(n1[i].split()), set(n2[i].split())
    p1 = {x[:4] for x in t1}
    p2 = {x[:4] for x in t2}
    ta = set(a1[i].split()) - {"unit"}
    tb = set(a2[i].split()) - {"unit"}
    print("-" * 104)
    print(f"NAME {n1[i]!r} || {n2[i]!r}   tok={t1&t2} pre={p1&p2}")
    print(f"ADDR {a1[i]!r}")
    print(f"     {a2[i]!r}   addr_tok={ta&tb} hn={h1[i]}/{h2[i]} zip={z1[i]}/{z2[i]}")
