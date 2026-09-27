"""Score the recovery candidates and measure the merged, honest macro F0.5.

The baseline predictions of the first pass are fixed; the only free parameter
is the threshold applied to the recovery candidates.  Because the two entity
sets are disjoint, the merged score of a validation entity is a simple sum, so
the whole sweep is one descending pass over the recovery scores.

Usage:  python 08b_recover_score.py --split train --tag _recv [--base-thr 0.669231]
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
M6 = importlib.import_module("06_model")
import common
from common import WORK, Right, Source


def sweep_merged(bl, by, bs, rl, ry, rs, n_val, ntrue, base_thr):
    """Descending pass over recovery scores; returns (best_thr, best_macro, curve)."""
    have = bs >= base_thr
    npred_b = np.bincount(bl[have], minlength=n_val).astype(np.int64)
    tp_b = np.bincount(bl[have], weights=by[have], minlength=n_val)
    order = np.argsort(-rs, kind="stable")
    rl, ry, rs = rl[order], ry[order], rs[order]
    qs = np.unique(np.round(np.concatenate([
        np.linspace(0.9990, 0.99999, 40),
        1.0 - np.geomspace(1e-6, 0.5, 60),
        np.linspace(0.3, 0.95, 30)]), 8))
    best = (None, -1.0, None)
    curve = []
    pos = 0
    npred_r = np.zeros(n_val, np.int64)
    tp_r = np.zeros(n_val, np.float64)
    for thr in qs[::-1]:
        k = int(np.searchsorted(-rs, -thr, side="right"))
        if k <= pos:
            continue
        sl = slice(pos, k)
        npred_r += np.bincount(rl[sl], minlength=n_val)
        tp_r += np.bincount(rl[sl], weights=ry[sl], minlength=n_val)
        pos = k
        f = common.entity_fbeta(tp_b + tp_r, npred_b + npred_r, ntrue)
        m = float(f.mean())
        curve.append((float(thr), m, int((npred_b + npred_r).sum()),
                      int(round((tp_b + tp_r).sum()))))
        if m > best[1]:
            best = (float(thr), m, (int((npred_b + npred_r).sum()),
                                    int(round((tp_b + tp_r).sum()))))
    return best, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="_recv")
    args = ap.parse_args()
    split, tag = args.split, args.tag
    with open(os.path.join(WORK, f"model2_{split}.pkl"), "rb") as f:
        mdl = pickle.load(f)
    base_thr = float(mdl["thr"])
    d = M6.load(split, keys=("l", "r", "y", "sv"))
    rec = {k: np.load(os.path.join(WORK, f"pairs_{split}{tag}_{k}.npy"), mmap_mode="r")
           for k in ("l", "r", "y", "f", "fz")}
    L = Source(f"{split}_s1")
    R = Right(split)
    n_all = len(L)

    rng = np.random.default_rng(M6.SEED)
    is_valid = np.zeros(n_all, bool)
    is_valid[rng.choice(n_all, int(n_all * M6.VALID_FRAC), replace=False)] = True
    ents = np.flatnonzero(is_valid)
    remap = np.full(n_all, -1, np.int64)
    remap[ents] = np.arange(len(ents))
    n_val = len(ents)

    l = np.asarray(d["l"]).astype(np.int64)
    y = np.asarray(d["y"]).astype(np.float64)
    sv = np.asarray(d["sv"]).astype(np.float64)
    va = is_valid[l]
    bl, by, bs = remap[l[va]], y[va], sv

    rl = np.asarray(rec["l"]).astype(np.int64)
    rr = np.asarray(rec["r"]).astype(np.int64)
    ry = np.asarray(rec["y"]).astype(np.float64)
    in_val = is_valid[rl]
    print(f"recovery pairs={len(rl):,}  in validation entities={int(in_val.sum()):,}  "
          f"positives={int(ry.sum()):,}", flush=True)
    rs = np.empty(int(in_val.sum()), np.float64)
    idx = np.flatnonzero(in_val)
    nf = rec["f"].shape[1]
    for b0 in range(0, len(idx), 2_000_000):
        b1 = min(b0 + 2_000_000, len(idx))
        sel = idx[b0:b1]
        X = np.empty((b1 - b0, nf + rec["fz"].shape[1]), np.float32)
        X[:, :nf] = rec["f"][sel]
        X[:, nf:] = rec["fz"][sel]
        rs[b0:b1] = mdl["model"].predict_proba(X)[:, 1]
    rl, ry, rr = remap[rl[idx]], ry[idx], rr[idx]

    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    tl, tr = remap[tl[mm]], tr[mm]
    ntrue = np.bincount(tl, minlength=n_val).astype(np.int64)

    # baseline only
    m0 = common.score(bl[bs >= base_thr], np.asarray(d["r"]).astype(np.int64)[va][bs >= base_thr],
                      tl, tr, n_val, R.n)
    print(common.fmt("BASELINE", m0), flush=True)
    cov = int(ry.sum())
    print(f"missed pairs of the validation entities recovered by the new families: "
          f"{cov:,}", flush=True)

    (thr, macro, extra), curve = sweep_merged(bl, by, bs, rl, ry, rs, n_val,
                                              ntrue, base_thr)
    pr_r = rr[rs >= thr]
    keep_b = bs >= base_thr
    m = common.score(np.concatenate([bl[keep_b], rl[rs >= thr]]),
                     np.concatenate([np.asarray(d["r"]).astype(np.int64)[va][keep_b], pr_r]),
                     tl, tr, n_val, R.n)
    print(f"\nMERGED recovery threshold={thr:.6f} macro_f0.5={macro:.5f} "
          f"(baseline {m0['macro_fbeta']:.5f}, delta {macro-m0['macro_fbeta']:+.5f})")
    print(common.fmt("  MERGED", m))
    print("\ncurve (rec_thr, macro, pred, tp):")
    for c in curve[::max(len(curve) // 15, 1)]:
        print(f"   {c[0]:.6f} {c[1]:.5f} {c[2]:>9,} {c[3]:>9,}")
    np.save(os.path.join(WORK, f"pairs_{split}{tag}_sv.npy"), rs.astype(np.float32))
    with open(os.path.join(WORK, f"recover_{split}_thr.txt"), "w") as f:
        f.write(f"base_thr={base_thr}\nrec_thr={thr}\nmacro={macro}\n"
                f"baseline={m0['macro_fbeta']}\n")


if __name__ == "__main__":
    main()
