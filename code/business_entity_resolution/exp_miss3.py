"""Honest stage-2 score against FULL ground truth, and who owns the missed pairs.

Two things 06b_threshold.py does not measure:

1. It builds the per-entity true-pair count from the candidate labels, so a
   validation entity whose true match never became a candidate is scored as a
   perfect singleton.  With candidate coverage ~0.85 that credits ~15% of the
   non-singleton entities a free 1.0, which is exactly the loss the leaderboard
   will charge us.  Here the truth comes from load_gt instead.
2. The missed pairs are attributed to entities by their current prediction
   status, which sizes the opportunity of a second-chance pass restricted to
   entities that currently predict nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
M6 = importlib.import_module("06_model")
import common
from common import WORK, Right, Source

SEED = M6.SEED
VALID_FRAC = M6.VALID_FRAC


def main():
    split = "train"
    d = M6.load(split, keys=("l", "r", "y", "sv"))
    L = Source(f"{split}_s1")
    R = Right(split)
    l = np.asarray(d["l"]).astype(np.int64)
    r = np.asarray(d["r"]).astype(np.int64)
    y = np.asarray(d["y"]).astype(np.int8)
    sv = np.asarray(d["sv"]).astype(np.float64)
    n_all = len(L)

    rng = np.random.default_rng(SEED)
    is_valid = np.zeros(n_all, bool)
    is_valid[rng.choice(n_all, int(n_all * VALID_FRAC), replace=False)] = True
    ents = np.flatnonzero(is_valid)
    remap = np.full(n_all, -1, np.int64)
    remap[ents] = np.arange(len(ents))
    n_val = len(ents)

    va_idx = np.flatnonzero(is_valid[l])
    lv, rv, yv, svv = remap[l[va_idx]], r[va_idx], y[va_idx], sv
    ntrue_cand = np.bincount(lv, weights=yv, minlength=n_val).astype(np.int64)

    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    tl, tr = remap[tl[mm]], tr[mm]
    ntrue = np.bincount(tl, minlength=n_val).astype(np.int64)
    print(f"validation entities={n_val:,}  candidate pairs={len(lv):,}")
    print(f"  true pairs of validation entities = {len(tl):,}")
    print(f"  of those, present as candidates    = {int(yv.sum()):,} "
          f"({yv.sum()/len(tl):.4f})")
    print(f"  true singletons (full truth)       = {int((ntrue == 0).sum()):,}")
    print(f"  apparent singletons (from labels)  = {int((ntrue_cand == 0).sum()):,}")

    (thr, macro, extra), curve = M6.sweep(lv, yv.astype(np.float64), svv, n_val, ntrue)
    pl, pr = lv[svv >= thr], rv[svv >= thr]
    m = common.score(pl, pr, tl, tr, n_val, R.n)
    print(f"\nFULL-TRUTH held-out: threshold={thr:.6f} macro_f0.5={macro:.5f}")
    print(common.fmt("  FULL-TRUTH", m))
    m_old = common.score(remap[l[va_idx][sv >= 0.669231]],
                         r[va_idx][sv >= 0.669231], tl, tr, n_val, R.n)
    print(common.fmt("  @0.669231", m_old))

    # who owns the missed pairs?
    have = np.bincount(pl, minlength=n_val)
    cov = np.zeros(n_val, np.int64)
    np.add.at(cov, tl, 1)
    got = np.zeros(n_val, np.int64)
    np.add.at(got, lv[yv == 1], 1)
    miss_per = cov - got
    print(f"\nmissed true pairs = {int(miss_per.sum()):,} over {int((miss_per>0).sum()):,} entities")
    for name, mask in (("left already predicts", have > 0),
                       ("left predicts nothing", have == 0)):
        print(f"  {name:<22s} entities={int(mask.sum()):>7,} "
              f"missed pairs={int(miss_per[mask].sum()):>9,} "
              f"({miss_per[mask].sum()/max(miss_per.sum(),1):.4f})")
    sing = ntrue == 0
    print(f"  of the missed-pair entities, true singletons = {int((miss_per>0)[sing].sum()):,}"
          f" (their F is already 0.0 only if they predict; "
          f"{int(((miss_per>0)&sing&(have>0)).sum()):,} do)")
    print(f"  singleton entities with a false match = {int((sing & (have>0)).sum()):,}")

    for c in curve[::max(len(curve) // 12, 1)]:
        print(f"   thr={c[0]:.6f} macro={c[1]:.5f} pred={c[2]:>9,} tp={c[3]:>9,}")


if __name__ == "__main__":
    main()
