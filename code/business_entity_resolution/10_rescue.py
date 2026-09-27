"""Rescue rule: for entities the first pass leaves empty, accept their best candidate.

A per-entity F0.5 with the singleton rule gives an entity with no prediction and
at least one true match a hard 0.  On the held-out split that is 14,894
entities (6.8% of all validation entities) -- by far the cheapest macro left to
win, and it needs no new candidates at all: those entities already have scored
candidates, the first pass just rejected every one of them.

The rule is deliberately one-sided:

    predicted(e) = {p : score(p) >= base_thr}
                   if that set is empty and best_score(e) >= rescue_thr
                   = {best candidate of e}
                   otherwise unchanged

so it can only turn a 0 into something, never change an existing prediction.
Both thresholds are swept against the full ground truth.
"""
import argparse
import importlib
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M6 = importlib.import_module("06_model")
import common
from common import WORK, Right, Source


def macro_at(ntrue, npred, tp, f0_5=0.5):
    return common.entity_fbeta(tp, npred, ntrue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="also merge this recovery tag")
    ap.add_argument("--rec-thr", type=float, default=0.0)
    args = ap.parse_args()
    split = "train"
    d = M6.load(split, keys=("l", "r", "y", "sv"))
    L = Source(f"{split}_s1")
    R = Right(split)
    n_all = len(L)
    with open(os.path.join(WORK, "model2_train.pkl"), "rb") as f:
        base_thr = float(pickle.load(f)["thr"])
    l = np.asarray(d["l"]).astype(np.int64)
    y = np.asarray(d["y"]).astype(np.float64)
    sv = np.asarray(d["sv"]).astype(np.float64)
    rng = np.random.default_rng(M6.SEED)
    is_valid = np.zeros(n_all, bool)
    is_valid[rng.choice(n_all, int(n_all * M6.VALID_FRAC), replace=False)] = True
    ents = np.flatnonzero(is_valid)
    remap = np.full(n_all, -1, np.int64)
    remap[ents] = np.arange(len(ents))
    n_val = len(ents)
    va = is_valid[l]
    bl, by, bs = remap[l[va]], y[va], sv

    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    tl, tr = remap[tl[mm]], tr[mm]
    ntrue = np.bincount(tl, minlength=n_val).astype(np.int64)

    keep = bs >= base_thr
    npred = np.bincount(bl[keep], minlength=n_val).astype(np.int64)
    tp = np.bincount(bl[keep], weights=by[keep], minlength=n_val)
    m0 = float(common.entity_fbeta(tp, npred, ntrue).mean())
    print(f"baseline  thr={base_thr:.6f}  macro={m0:.5f}  "
          f"pred={int(npred.sum()):,}  tp={int(round(tp.sum())):,}", flush=True)

    # recovery pool (optional): the extra candidates from 08_recover.py
    if args.tag:
        rl = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{args.tag}_l.npy")))
        rr = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{args.tag}_r.npy")))
        ry = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{args.tag}_y.npy")))
        rs = np.load(os.path.join(WORK, f"pairs_{split}{args.tag}_sv.npy"))
        r_va = is_valid[rl]
        rl, rr, ry, rs = remap[rl[r_va]], rr[r_va], ry[r_va], rs[r_va]
        rk = rs >= args.rec_thr
        add = ~np.isin(rl[rk] * (R.n + 1) + rr[rk],
                       bl[keep] * (R.n + 1) + np.asarray(d["r"])[va][keep].astype(np.int64))
        npred = npred + np.bincount(rl[rk][add], minlength=n_val).astype(np.int64)
        tp = tp + np.bincount(rl[rk][add], weights=ry[rk][add], minlength=n_val)
        m0 = float(common.entity_fbeta(tp, npred, ntrue).mean())
        print(f"+recovery {args.tag} @ {args.rec_thr:.6f}: macro={m0:.5f} "
              f"pred={int(npred.sum()):,} tp={int(round(tp.sum())):,}", flush=True)
        pool_l = np.concatenate([bl, rl])
        pool_y = np.concatenate([by, ry])
        pool_s = np.concatenate([bs, rs])
    else:
        pool_l, pool_y, pool_s = bl, by, bs

    # best candidate per entity over the whole pool (baseline + recovery)
    order = np.argsort(pool_l * 2.0 - pool_s)  # placeholder, replaced below
    del order
    best_s = np.full(n_val, -1.0)
    best_tp = np.zeros(n_val, bool)
    np.maximum.at(best_s, pool_l, pool_s)
    hit = pool_s == best_s[pool_l]
    best_tp[pool_l[hit & (pool_y == 1)]] = True
    empty = npred == 0
    print(f"\nempty entities: {int(empty.sum()):,}; of those, "
          f"{int((ntrue[empty]>0).sum()):,} have a true match and "
          f"{int((best_tp & empty).sum()):,} have it as their best candidate", flush=True)
    best_all = float(common.entity_fbeta(tp, npred, ntrue).mean())
    print(f"{'rescue_thr':>11} {'macro':>8} {'delta':>8} {'rescued':>8} {'true':>7} {'false':>7}")
    for rt in [0.999999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
               0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
        fire = empty & (best_s >= rt) & (best_s > 0)
        npred_r = npred + fire.astype(np.int64)
        tp_r = tp + fire * best_tp
        m = float(common.entity_fbeta(tp_r, npred_r, ntrue).mean())
        good = int((fire & best_tp).sum())
        print(f"{rt:>11.6f} {m:>8.5f} {m-best_all:>+8.5f} {int(fire.sum()):>8,} "
              f"{good:>7,} {int(fire.sum())-good:>7,}")


if __name__ == "__main__":
    main()
