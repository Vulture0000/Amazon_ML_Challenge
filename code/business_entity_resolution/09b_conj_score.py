"""Honest merged scoring for the conjunctive pass (entity sets overlap).

Unlike the recovery pass, which only touched entities the first pass left empty,
the conjunctive pass runs over *every* held-out entity, so its candidates can
duplicate a pair the first pass already predicted.  A naive merge would then
double-count those pairs and inflate the metric, so every (left, right) pair is
deduplicated across the pools before the threshold sweep, and the sweep
maximises the true macro F0.5 over the full ground truth.

The first pass keeps its own operating point; the extra pools share one
threshold that is swept on the merged objective.

Usage:
  python 09b_conj_score.py --tag _conj --rec-tag _recv2 --rec-thr 0.794601
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


def score_pool(split, tag, mdl, nf, is_valid):
    """Load a candidate pool and stage-2 score it (cached in pairs_{split}{tag}_sv)."""
    path = os.path.join(WORK, f"pairs_{split}{tag}_sv.npy")
    l = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{tag}_l.npy")))
    r = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{tag}_r.npy")))
    y = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{tag}_y.npy")), np.float64)
    if os.path.exists(path):
        s = np.asarray(np.load(path), np.float64)
    else:
        d = {"l": np.load(os.path.join(WORK, f"pairs_{split}{tag}_l.npy"), mmap_mode="r"),
             "f": np.load(os.path.join(WORK, f"pairs_{split}{tag}_f.npy"), mmap_mode="r"),
             "fz": np.load(os.path.join(WORK, f"pairs_{split}{tag}_fz.npy"), mmap_mode="r")}
        s = np.empty(len(l), np.float64)
        for b0 in range(0, len(l), 2_000_000):
            b1 = min(b0 + 2_000_000, len(l))
            s[b0:b1] = mdl["model"].predict_proba(
                M6.gather(d, np.arange(b0, b1), nf))[:, 1]
        np.save(path, s.astype(np.float32))
    k = is_valid[l]
    return l[k].astype(np.int64), r[k].astype(np.int64), y[k], s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_conj")
    ap.add_argument("--rec-tag", default=None)
    ap.add_argument("--rec-thr", type=float, default=0.0)
    args = ap.parse_args()
    split = "train"
    L, R = Source(f"{split}_s1"), Right(split)
    n_all = len(L)
    with open(os.path.join(WORK, "model2_train.pkl"), "rb") as f:
        mdl = pickle.load(f)
    base_thr = float(mdl["thr"])
    d = M6.load(split, keys=("l", "r", "y", "f", "sv"))
    nf = d["f"].shape[1]
    l = np.asarray(d["l"]).astype(np.int64)
    r = np.asarray(d["r"]).astype(np.int64)
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
    pools = [(remap[l[va]], r[va], y[va], sv, "base")]
    print(f"pool base       {len(pools[0][0]):>10,} pairs "
          f"{int(pools[0][2].sum()):>9,} positives", flush=True)
    if args.rec_tag:
        p = score_pool(split, args.rec_tag, mdl, nf, is_valid)
        p = (remap[p[0]], p[1], p[2], p[3], f"rec:{args.rec_tag}")
        pools.append(p)
        print(f"pool {p[4]:<12s}{len(p[0]):>10,} pairs {int(p[2].sum()):>9,} positives",
              flush=True)
    p = score_pool(split, args.tag, mdl, nf, is_valid)
    p = (remap[p[0]], p[1], p[2], p[3], f"conj:{args.tag}")
    pools.append(p)
    print(f"pool {p[4]:<12s}{len(p[0]):>10,} pairs {int(p[2].sum()):>9,} positives",
          flush=True)

    pl = np.concatenate([p[0] for p in pools])
    pr = np.concatenate([p[1] for p in pools])
    py = np.concatenate([p[2] for p in pools])
    ps = np.concatenate([p[3] for p in pools])
    is_extra = np.concatenate([np.zeros(len(p[0]), bool) if p[4] == "base"
                               else np.ones(len(p[0]), bool) for p in pools])
    # (left, right) must be packed in a field wider than R.n: 2**7 > R.n here,
    # so a 10**7 multiplier silently collided and dropped real pairs
    assert R.n <= (1 << 24), "the dedup key would collide"
    k = (pl << 24) | pr
    _, idx = np.unique(k, return_index=True)
    idx = np.sort(idx)
    pl, pr, py, ps, is_extra = pl[idx], pr[idx], py[idx], ps[idx], is_extra[idx]
    print(f"merged: {len(pl):,} unique pairs, {int(py.sum()):,} positives "
          f"({int(py.sum())-int(sum(p[2].sum() for p in pools)):,} recovered by dedup)",
          flush=True)

    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    tl, tr = remap[tl[mm]], tr[mm]
    ntrue = np.bincount(tl, minlength=n_val).astype(np.int64)

    o = np.argsort(-ps, kind="stable")
    pl, pr, py, ps, is_extra = pl[o], pr[o], py[o], ps[o], is_extra[o]
    sel_b = ~is_extra & (ps >= base_thr)
    npred_b = np.bincount(pl[sel_b], minlength=n_val).astype(np.int64)
    tp_b = np.bincount(pl[sel_b], weights=py[sel_b], minlength=n_val)
    m0 = float(common.entity_fbeta(tp_b, npred_b, ntrue).mean())
    print(f"\nbaseline thr={base_thr:.6f} macro={m0:.5f} pred={int(npred_b.sum()):,} "
          f"tp={int(round(tp_b.sum())):,}\n")
    print(f"{'extra_thr':>10} {'macro':>8} {'delta':>8} {'pred':>10} {'tp':>10}")
    qs = np.unique(np.round(np.concatenate([
        np.linspace(0.9990, 0.99999, 30), 1.0 - np.geomspace(1e-6, 0.5, 70),
        np.linspace(0.3, 0.95, 40)]), 8))
    best = None
    for thr in qs:
        use = (ps >= max(thr, base_thr)) | (~is_extra & (ps >= base_thr))
        npred = np.bincount(pl[use], minlength=n_val).astype(np.int64)
        tp = np.bincount(pl[use], weights=py[use], minlength=n_val)
        m = float(common.entity_fbeta(tp, npred, ntrue).mean())
        if m > m0 and (best is None or m > best[1]):
            best = (float(thr), m, thr)
        print(f"{thr:>10.6f} {m:>8.5f} {m-m0:>+8.5f} {int(npred.sum()):>10,} "
              f"{int(round(tp.sum())):>10,}")
    if best:
        thr, macro, _ = best
        use = (ps >= max(thr, base_thr)) | (~is_extra & (ps >= base_thr))
        print()
        print(common.fmt("MERGED", common.score(pl[use], pr[use], tl, tr, n_val, R.n)))
        with open(os.path.join(WORK, "conj_best_thr.txt"), "w") as f:
            f.write(f"base_thr={base_thr}\nextra_thr={thr}\nmacro={macro}\n"
                    f"baseline={m0}\ntag={args.tag}\n")
        np.save(os.path.join(WORK, f"pairs_{split}{args.tag}_sel.npy"), use)
        print(f"BEST extra_thr={thr:.6f} macro={macro:.5f} "
              f"(delta {macro-m0:+.5f}) -> work/conj_best_thr.txt")


if __name__ == "__main__":
    main()
