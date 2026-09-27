"""Per-entity decision layer, cross-fitted on the held-out split.

A single global threshold is a coarse instrument: it cannot express that an
entity whose best candidate scores 0.99 with a runner-up at 0.70 probably has
two matches, while an entity whose only candidate is at 0.70 probably has none
(the singleton rule makes a wrong prediction cost a full 1.0).  The layer keeps
the ranking the model produces and decides, per entity, *how many* of its
candidates to accept:

    rank 1        accepted if score >= t1
    ranks 2..8    accepted if score >= t2 and the best score >= t3

which subsumes the current rule (t2 = t1 = t3) and adds a per-entity gate.
(t1, t2, t3) are fitted on one half of the held-out entities and scored on the
other half, and vice versa, so the reported number involves no fitting on the
rows it is measured on.  It also prints where the remaining macro is lost.

Usage:  python 12_decide.py --tag _conj --rec-tag _recv2 --rec-thr 0.794601
"""
import argparse
import importlib
import itertools
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M6 = importlib.import_module("06_model")
import common
from common import WORK, Right, Source

TOPK = 8


def macro_src(S, Y, ntrue, is_extra, t_base, t_extra):
    use = is_extra & (S >= t_extra)
    use |= ~is_extra & (S >= t_base)
    npred = use.sum(1).astype(np.float64)
    tp = (use * Y).sum(1)
    return float(common.entity_fbeta(tp, npred, ntrue).mean())


def build(pool_l, pool_r, pool_y, pool_s, extra, n_val, R):
    """Top-TOPK candidates per entity, as dense (score, label) matrices."""
    o = np.lexsort((-pool_s, pool_l))
    l, s, y = pool_l[o], pool_s[o], pool_y[o]
    st = np.flatnonzero(np.concatenate(([True], l[1:] != l[:-1])))
    rank = np.arange(len(l)) - np.repeat(st, np.diff(np.append(st, len(l))))
    keep = rank < TOPK
    S = np.zeros((n_val, TOPK), np.float32)
    Y = np.zeros((n_val, TOPK), np.float32)
    E = np.zeros((n_val, TOPK), bool)
    S[l[keep], rank[keep]] = s[keep]
    Y[l[keep], rank[keep]] = y[keep]
    E[l[keep], rank[keep]] = extra[o][keep]
    return S, Y, E


def macro(S, Y, ntrue, t1, t2, t3):
    p1 = S[:, 0] >= t1
    pr = (S >= t2) & (S[:, 0:1] >= t3)
    pred = pr.copy()
    pred[:, 0] = p1
    npred = pred.sum(1).astype(np.float64)
    tp = (pred * Y).sum(1)
    return float(common.entity_fbeta(tp, npred, ntrue).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_conj")
    ap.add_argument("--rec-tag", default="_recv2")
    ap.add_argument("--rec-thr", type=float, default=0.794601)
    args = ap.parse_args()
    split = "train"
    L, R = Source(f"{split}_s1"), Right(split)
    n_all = len(L)
    with open(os.path.join(WORK, "model2_train.pkl"), "rb") as f:
        base_thr = float(pickle.load(f)["thr"])
    d = M6.load(split, keys=("l", "r", "y", "sv"))
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
    pools = [(remap[l[is_valid[l]]], r[is_valid[l]], y[is_valid[l]], sv,
              np.zeros(int(is_valid[l].sum()), bool))]
    for tag in (args.rec_tag, args.tag):
        pl = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{tag}_l.npy")))
        pr = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{tag}_r.npy")))
        py = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{tag}_y.npy")), np.float64)
        ps = np.asarray(np.load(os.path.join(WORK, f"pairs_{split}{tag}_sv.npy")),
                        np.float64)
        k = is_valid[pl]
        pools.append((remap[pl[k]], pr[k], py[k], ps[k],
                      np.ones(int(k.sum()), bool)))
    pl = np.concatenate([p[0] for p in pools])
    pr = np.concatenate([p[1] for p in pools])
    py = np.concatenate([p[2] for p in pools])
    ps = np.concatenate([p[3] for p in pools])
    pe = np.concatenate([p[4] for p in pools])
    kk = pl * (10 ** 7) + pr
    _, idx = np.unique(kk, return_index=True)
    idx = np.sort(idx)
    pl, pr, py, ps, pe = pl[idx], pr[idx], py[idx], ps[idx], pe[idx]
    print(f"merged pool {len(pl):,} pairs, {int(py.sum()):,} positives", flush=True)

    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    tl, tr = remap[tl[mm]], tr[mm]
    ntrue = np.bincount(tl, minlength=n_val).astype(np.int64)
    S, Y, E = build(pl, pr, py, ps, pe, n_val, R)
    print(f"entities whose best candidate score >= base_thr: "
          f"{int((S[:,0]>=base_thr).sum()):,}", flush=True)

    grid1 = [0.999, 0.995, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
             0.55, 0.5, 0.4, 0.3]
    grid2 = [0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    grid3 = [0.0, 0.5, 0.9, 0.95, 0.99, 0.999]
    fold = (np.arange(n_val) % 2)
    fits = []
    for f in (0, 1):
        trn, tst = fold == f, fold != f
        best = (-1.0, None)
        for t1, t2, t3 in itertools.product(grid1, grid2, grid3):
            m = macro(S[trn], Y[trn], ntrue[trn], t1, t2, t3)
            if m > best[0]:
                best = (m, (t1, t2, t3))
        held = macro(S[tst], Y[tst], ntrue[tst], *best[1])
        fits.append((best[1], best[0], held))
        print(f"fold {f}: fitted {best[1]} on {int(trn.sum()):,} rows "
              f"(train macro {best[0]:.5f}) -> held-out {held:.5f}", flush=True)
    cf_macro = float(np.mean([h for _, _, h in fits]))
    # reference points on the same rows
    ref_single = macro(S, Y, ntrue, base_thr, base_thr, 0.0)
    ref_src = max((macro_src(S, Y, ntrue, E, base_thr, t), t)
                  for t in np.arange(0.30, 0.999, 0.005))
    print(f"\nsingle global threshold {base_thr:.6f}: macro={ref_single:.5f}")
    print(f"per-source thresholds (base {base_thr:.6f}, extra {ref_src[1]:.3f}): "
          f"macro={ref_src[0]:.5f}")
    ref_single = ref_src[0]
    print(f"cross-fitted decision layer: macro={cf_macro:.5f} "
          f"({cf_macro-ref_single:+.5f})")
    print(f"decision layer vs the per-source rule: {cf_macro-ref_single:+.5f}")

    # where the remaining macro is lost, under the per-source rule
    prd = (E & (S >= ref_src[1])) | (~E & (S >= base_thr))
    npred = prd.sum(1)
    tp = (prd * Y).sum(1)
    f = common.entity_fbeta(tp, npred, ntrue)
    sing = ntrue == 0
    empty = npred == 0
    print(f"\nloss decomposition over {n_val:,} entities "
          f"(per-source rule, base {base_thr:.4f} / extra {ref_src[1]:.3f}):")
    print(f"  true singletons, left empty   : {int((sing & empty).sum()):>7,} "
          f"mean f={f[sing & empty].mean() if (sing & empty).any() else 1:.4f}  "
          f"loss={1 - f[sing & empty].sum()/n_val:+.4f}")
    print(f"  true singletons, wrongly filled: {int((sing & ~empty).sum()):>7,} "
          f"mean f={f[sing & ~empty].mean() if (sing & ~empty).any() else 0:.4f}  "
          f"loss={f[sing & ~empty].sum()/n_val:.4f}")
    m = ~sing
    print(f"  matched entities, perfect      : {int((m & (f > 0.999)).sum()):>7,}  "
          f"loss={0.0:.4f}")
    part = m & (f > 0) & (f <= 0.999)
    print(f"  matched entities, partial      : {int(part.sum()):>7,}  "
          f"mean f={f[part].mean():.4f}  loss={1 - f[part].sum()/n_val:.4f}")
    # of the entities that score zero, how many are unrecoverable in principle?
    zero = m & (f == 0)
    inpool = (Y.sum(1) > 0)
    print(f"    of those, true pair present in pool: {int((zero & inpool).sum()):,} "
          f"(ranking problem)   not in pool: {int((zero & ~inpool).sum()):,} "
          f"(blocking problem)")
    print(f"  matched entities, nothing      : {int((m & (f == 0)).sum()):>7,}  "
          f"loss={1 - f[m & (f == 0)].sum()/n_val:.4f}")
    print(f"  -> mean f = {f.mean():.5f}")
    with open(os.path.join(WORK, "decide_params.txt"), "w") as fh:
        fh.write(f"t1={fits[0][0][0]}\nt2={fits[0][0][1]}\nt3={fits[0][0][2]}\n"
                 f"cross_fitted_macro={cf_macro}\nsingle_thr_macro={ref_single}\n")


if __name__ == "__main__":
    main()
