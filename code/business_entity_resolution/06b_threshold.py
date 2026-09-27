"""Recompute the stage-2 threshold and the honest held-out macro F0.5.

The validation score in 06_model.py was reported over all split entities, which
credits ~2M entities that are not in the validation subset as perfect singletons
and therefore inflates the macro average.  This script restricts the metric to
the validation entities themselves (remapping their left rows).

The truth used here is the *full* ground truth from common.load_gt(), not the
candidate labels.  Deriving the truth from the candidate labels treats every
validation entity whose true pairs were never proposed as a perfect singleton,
which inflated the earlier report to 0.95393; the real number is 0.85320.
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
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
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
    va_idx = np.flatnonzero(is_valid[l])
    lv, rv, yv, svv = l[va_idx], r[va_idx], y[va_idx], sv
    ents = np.flatnonzero(is_valid)
    remap = np.full(n_all, -1, np.int64)
    remap[ents] = np.arange(len(ents))
    lv = remap[lv]
    n_val = len(ents)
    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    ntrue = np.bincount(remap[tl[mm]], minlength=n_val).astype(np.int64)
    print(f"validation: {n_val:,} entities, {len(lv):,} pairs, "
          f"{int(yv.sum()):,} candidate positives", flush=True)
    print(f"  true pairs = {int(ntrue.sum()):,} over {int((ntrue>0).sum()):,} entities",
          flush=True)
    print(f"  true singletons in validation = {int((ntrue==0).sum()):,}", flush=True)
    print(f"  candidate coverage of true pairs = "
          f"{yv.sum() / max(int(ntrue.sum()),1):.4f}", flush=True)
    (thr, macro, extra), curve = M6.sweep(lv, yv.astype(np.float64), svv,
                                          n_val, ntrue)
    m = common.score(lv[svv >= thr], rv[svv >= thr], remap[tl[mm]], tr[mm],
                     n_val, R.n)
    print(f"\nHONEST held-out: threshold={thr:.6f} macro_f0.5={macro:.5f}")
    print(common.fmt("  HONEST", m))
    print("\ncurve (thr, macro, pred, tp):")
    for c in curve[::max(len(curve) // 15, 1)]:
        print(f"   {c[0]:.6f} {c[1]:.5f} {c[2]:>9,} {c[3]:>9,}")
    return thr


if __name__ == "__main__":
    main()
