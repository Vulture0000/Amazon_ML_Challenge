"""Measured comparison of stage-2 feature sets on the hard cases.

After the two recovery passes the pooled candidate set already contains 94% of
the true pairs at 0.906 precision and 67% of the remaining macro loss sits in
entities that are matched *partially* (78,415 entities, mean f=0.861), so the
binding constraint is ranking multi-match entities -- not retrieval, not the
threshold.

Three families of signal are available, and the third is the interesting one:

  cheap    tfidf/tfidf/rapidfuzz similarity, lengths, country, house/zip
  global   crowd_r (how many entities claim this right record), crowd_l (how
           many candidates this entity has), rank1 (rank by stage-1 score),
           is_s2 (right record from S2 not S3)
  compete  the one-to-one prior: how strongly this entity claims a right
           record that other entities also claim, the rank of this claim among
           claimants, the margin over the entity's next best candidate of the
           same source, and whether the pair survived the first pass.  No
           pairwise string comparison can express these; they are computed
           from the stage-1 score plus a first pass of the stage-2 score.

Every set is trained on the same rows with the same budget and compared by
average precision on the held-out entity split, so the answer is measured.
"""
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M6 = importlib.import_module("06_model")
import common
from common import WORK, Right, Source

N_CHUNKS = int(os.environ.get("N_CHUNKS", "8"))
NEG_PER = int(os.environ.get("NEG_PER", "150000"))
ITERS = int(os.environ.get("M2_ITER", "300"))
SEED = M6.SEED


def first_of(key, score, n):
    """max score per key, and the rank of each row's key when ordered by score"""
    n = max(n, int(key.max()) + 1 if len(key) else 1)
    best = np.full(n, -1.0, np.float32)
    np.maximum.at(best, key, score)
    o = np.lexsort((-score, key))
    k = key[o]
    start = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
    rank = np.empty(len(key), np.float32)
    rank[o] = np.arange(len(key)) - np.repeat(start, np.diff(np.append(start, len(key))))
    return best, rank + 1.0


def global_feats(l, r, sc, R):
    out = np.empty((len(l), 4), np.float32)
    out[:, 0] = np.log1p(np.bincount(r, minlength=R.n).astype(np.float32)[r])
    out[:, 1] = np.log1p(np.bincount(l).astype(np.float32)[l])
    _, rank = first_of(l, sc, 0)
    out[:, 2] = np.log1p(rank)
    out[:, 3] = (r < R.n2).astype(np.float32)
    return out


def compete_feats(l, r, sc, sv, R, base_thr):
    """[claim margin, claim rank, claims, first-pass flag, ent margin, src margin]"""
    n = len(l)
    out = np.empty((n, 6), np.float32)
    best_r, rank_r = first_of(r, sv, R.n)
    out[:, 0] = sv - best_r[r]
    out[:, 1] = np.log1p(rank_r)
    out[:, 2] = np.log1p(np.bincount(r, minlength=R.n).astype(np.float32)[r])
    out[:, 3] = (sv >= base_thr).astype(np.float32)
    # best candidate of the same source, per entity
    src = (r >= R.n2).astype(np.int64)
    key2 = l * 2 + src
    best_s, _ = first_of(key2, sv, 0)
    out[:, 4] = sv - best_s[key2]
    # margin over this entity's next best candidate (different right record)
    best_l, _ = first_of(l, sv, 0)
    second = np.full(len(l), 0.0, np.float32)
    o = np.lexsort((-sv, l))
    lo = l[o]
    st = np.flatnonzero(np.concatenate(([True], lo[1:] != lo[:-1])))
    grp = np.repeat(st, np.diff(np.append(st, n)))
    has2 = np.bincount(grp) >= 2
    if has2.any():
        p1 = np.flatnonzero(has2)
        second[p1] = sv[o][np.minimum(grp[p1] + 1, n - 1)]
    out[:, 5] = second - sv
    return out


def build_xy(rows, d, nzf, extra, chunk=2_000_000):
    nzfz = d["fz"].shape[1]
    X = np.empty((len(rows), nzf + nzfz + extra.shape[1]), np.float32)
    for b0 in range(0, len(rows), chunk):
        b1 = min(b0 + chunk, len(rows))
        X[b1 - b1 + b0:b0 + chunk, :nzf + nzfz] = M6.gather(d, rows[b0:b1], nzf)
        X[b0:b1, nzf + nzfz:] = extra[b0:b1]
    return X


def main():
    t0 = time.time()
    L, R = Source("train_s1"), Right("train")
    n_all = len(L)
    d = M6.load("train", keys=("l", "r", "y", "sc", "f", "fz"))
    l = np.asarray(d["l"]).astype(np.int64)
    r = np.asarray(d["r"]).astype(np.int64)
    y = np.asarray(d["y"])
    sc = np.asarray(d["sc"]).astype(np.float32)
    sv = np.asarray(np.load(os.path.join(WORK, "pairs_train_sv_all.npy")))
    nzf = d["f"].shape[1]
    import pickle
    with open(os.path.join(WORK, "model2_train.pkl"), "rb") as fh:
        base_thr = float(pickle.load(fh)["thr"])
    rng = np.random.default_rng(SEED)
    is_valid = np.zeros(n_all, bool)
    is_valid[rng.choice(n_all, int(n_all * M6.VALID_FRAC), replace=False)] = True
    va = is_valid[l]
    va_idx = np.flatnonzero(va)
    print(f"pairs={len(l):,}  validation pairs={len(va_idx):,}  "
          f"base_thr={base_thr:.6f}  {time.time()-t0:.0f}s", flush=True)

    G = global_feats(l, r, sc, R)
    C = compete_feats(l, r, sc, sv, R, base_thr)
    sets = {
        "base39": np.zeros((len(l), 0), np.float32),
        "base39+global4": G,
        "base39+compete6": C,
        "base39+all10": np.concatenate([G, C], axis=1),
    }

    tr_pool = np.flatnonzero(~va)
    step = max(len(tr_pool) // N_CHUNKS, 1)
    picks = []
    for c in range(N_CHUNKS):
        sl = tr_pool[c * step:(c + 1) * step]
        if len(sl) == 0:
            break
        band = sl[np.isin(l[sl], np.unique(l[sl]))]
        pos = band[y[band] == 1]
        neg = band[y[band] == 0]
        picks.append(np.concatenate(
            [pos, rng.choice(neg, min(NEG_PER, len(neg)), replace=False)]))
    tr_idx = np.sort(np.concatenate(picks))
    print(f"training rows={len(tr_idx):,}  positives={int(y[tr_idx].sum()):,}", flush=True)

    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    ntrue_all = np.bincount(tl[mm], minlength=n_all).astype(np.int64)
    vent = np.flatnonzero(is_valid)
    n_val = len(vent)
    remap = np.full(n_all, -1, np.int64)
    remap[vent] = np.arange(n_val)
    yva = y[va_idx].astype(np.float64)
    cva = remap[l[va_idx]]
    tp_pool = np.bincount(cva[yva == 1], minlength=n_val).astype(np.int64)
    nt_pool = ntrue_all[vent]
    inpool = tp_pool > 0
    print(f"entities with >=1 true pair in the base pool: {int(inpool.sum()):,}", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    res = {}
    for name, extra in sets.items():
        t = time.time()
        Xtr = build_xy(tr_idx, d, nzf, extra[tr_idx])
        Xva = build_xy(va_idx, d, nzf, extra[va_idx])
        m = HistGradientBoostingClassifier(
            max_iter=ITERS, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.05, n_iter_no_change=15,
            random_state=SEED)
        m.fit(Xtr, y[tr_idx])
        p = m.predict_proba(Xva)[:, 1]
        ap = average_precision_score(y[va_idx], p)
        auc = roc_auc_score(y[va_idx], p)
        # macro over the entities that DO have a true match in the pool: this
        # isolates the ranking lever, which is where 67% of the loss sits, from
        # the retrieval lever, which the recovery pools already cover.
        yv = y[va_idx].astype(np.float64)
        lv = l[va_idx]
        best = (-1.0, 0.5)
        for t in np.arange(0.20, 0.999, 0.005):
            sel = p >= t
            npred = np.bincount(cva, weights=sel.astype(np.float64), minlength=n_val)
            tp = np.bincount(cva, weights=(sel * yva), minlength=n_val)
            f = common.entity_fbeta(tp, npred, nt_pool)
            m2 = f[inpool].mean()
            if m2 > best[0]:
                best = (m2, t)
        res[name] = ap
        print(f"  {name:<16s} AP={ap:.5f} AUC={auc:.5f} "
              f"macro|in-pool={best[0]:.5f} (thr {best[1]:.3f}) iters={m.n_iter_}", flush=True)
    b = res["base39"]
    print()
    for k, v in res.items():
        print(f"  {k:<16s} AP gain vs base39: {v-b:+.5f}")


if __name__ == "__main__":
    main()
