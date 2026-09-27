"""Train and validate the stage-2b competition ranker, then cache its scores.

Steps
  1. v1-score the recall pools (base pool scores already exist as sv_all)
  2. build the N_X2 competition features over the union of all pools
  3. fit on the same rows/budget as stage 2, with the extra columns appended
  4. re-sweep the threshold on the held-out entity split
  5. save v2 scores for every pool, so the sweep can use the merged pools

Run:  M2_ITER=400 python 14_train2b.py
"""
import importlib
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M6 = importlib.import_module("06_model")
S2 = importlib.import_module("stage2b")
import common
from common import WORK, Right, Source

TAGS = ["_recv2", "_conj"]
NEG_SAMPLE = float(os.environ.get("NEG_SAMPLE", M6.NEG_SAMPLE))
RECALL_NEG = float(os.environ.get("RECALL_NEG", "1.0"))
ITERS = int(os.environ.get("M2_ITER", "400"))


def rescore_only():
    """Re-score the recall pools with the saved model2b, no refit."""
    importlib.import_module("stage2b")
    split = "train"
    d = M6.load(split, keys=("f", "fz"))
    nzf = d["f"].shape[1]
    with open(os.path.join(WORK, f"model2b_{split}.pkl"), "rb") as fh:
        m2 = pickle.load(fh)
    for tag in TAGS:
        p = os.path.join(WORK, f"pairs_{split}{tag}_sv2.npy")
        if os.path.exists(p):
            os.remove(p)
        x2 = np.load(os.path.join(WORK, f"pairs_{split}{tag}_x2.npy"))
        s = S2.score(S2.pool_d(split, tag, d), np.arange(len(x2)), x2, nzf,
                     m2["model"], m2["n_x2"])
        np.save(p, s)
        print(f"{tag}: {len(s):,} rows rescored, mean {s.mean():.4f}", flush=True)


def main():
    if os.environ.get("RESCORE_ONLY"):
        rescore_only()
        return
    t0 = time.time()
    split = "train"
    L, R = Source(f"{split}_s1"), Right(split)
    n_all = len(L)
    d = M6.load(split, keys=("l", "r", "y", "sc", "f", "fz"))
    nzf = d["f"].shape[1]
    with open(os.path.join(WORK, f"model2_{split}.pkl"), "rb") as fh:
        v1 = pickle.load(fh)
    base_thr = float(v1["thr"])
    l = np.asarray(d["l"]).astype(np.int64)
    r = np.asarray(d["r"]).astype(np.int64)
    y = np.asarray(d["y"])
    sc = np.asarray(d["sc"]).astype(np.float32)
    print(f"base pool {len(l):,} pairs, base_thr={base_thr:.6f}", flush=True)

    print("step 1: v1 scores for the recall pools", flush=True)
    sv1 = [np.load(os.path.join(WORK, f"pairs_{split}_sv_all.npy"))]
    for tag in TAGS:
        sv1.append(S2.load_v1(split, tag, d, nzf, v1["model"]))
        print(f"  {tag}: {len(sv1[-1]):,} rows", flush=True)

    print("step 2: competition features over the union", flush=True)
    lu = np.concatenate([l] + [S2.pool(split, t)[0] for t in TAGS])
    ru = np.concatenate([r] + [S2.pool(split, t)[1] for t in TAGS])
    scu = np.concatenate([sc] + [S2.pool(split, t)[3] for t in TAGS])
    svu = np.concatenate(sv1)
    n_u = len(lu)
    bounds = np.append(np.cumsum([0] + [len(s) for s in sv1]), n_u)
    X2 = S2.extras(lu, ru, scu, svu, R, base_thr)
    del lu, ru, scu, svu
    xb = X2[bounds[0]:bounds[1]]
    np.save(os.path.join(WORK, f"pairs_{split}_x2.npy"), xb)
    for i, tag in enumerate(TAGS):
        np.save(os.path.join(WORK, f"pairs_{split}{tag}_x2.npy"),
                X2[bounds[i + 1]:bounds[i + 2]])
    print(f"  universe {n_u:,} rows, X2={X2.shape}, {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(M6.SEED)
    is_valid = np.zeros(n_all, bool)
    is_valid[rng.choice(n_all, int(n_all * M6.VALID_FRAC), replace=False)] = True
    va = is_valid[l]
    va_idx = np.flatnonzero(va)
    y_pool = [y]
    for tag in TAGS:
        y_pool.append(np.asarray(np.load(
            os.path.join(WORK, f"pairs_{split}{tag}_y.npy")), np.int8))
    l_pool = [l] + [S2.pool(split, t)[0] for t in TAGS]
    x2_pool = [xb] + [X2[bounds[i + 1]:bounds[i + 2]] for i in range(len(TAGS))]

    # Train on all three pools, not just the base pool.  The `first_pass` column
    # is v1's own decision, and inside the base pool that decision is nearly a
    # label (it is v1's swept operating point), so a model trained on the base
    # pool alone learns "v1 said no" as if it meant "not a match" -- and then
    # scores every recall-pool candidate as a non-match, which is precisely the
    # population those pools exist to rescue.  The recall pools are also the
    # only rows where the competition features take unusual values, since their
    # candidates are by construction ones the base blocking never proposed.
    sel = []
    for i in range(len(TAGS) + 1):
        lp, yp = l_pool[i], y_pool[i]
        m = ~is_valid[lp]
        pos = np.flatnonzero(m & (yp == 1))
        neg = np.flatnonzero(m & (yp == 0))
        frac = NEG_SAMPLE if i == 0 else RECALL_NEG
        if frac < 1.0:
            neg = rng.choice(neg, int(len(neg) * frac), replace=False)
        sel.append((i, np.concatenate([pos, neg])))
        print(f"  pool {i} ({'base' if i == 0 else TAGS[i-1]}): "
              f"{len(pos):,} positives + {len(neg):,} negatives", flush=True)
    total = sum(len(s) for _, s in sel)
    print(f"  train rows={total:,}  held-out base rows={len(va_idx):,}", flush=True)

    print("step 3: fit", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    model = HistGradientBoostingClassifier(
        max_iter=ITERS, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25, random_state=M6.SEED)
    t = time.time()
    ncol = nzf + d["fz"].shape[1] + S2.N_X2
    X = np.empty((total, ncol), np.float32)
    ytr = np.empty(total, np.int8)
    off = 0
    for i, rows in sel:
        dd = S2.pool_d(split, "" if i == 0 else TAGS[i - 1], d)
        nz = 0
        for b0 in range(0, len(rows), S2.CHUNK):
            b1 = min(b0 + S2.CHUNK, len(rows))
            n_ = b1 - b0
            X[off + b0:off + b1, :nz + dd["fz"].shape[1]] = M6.gather(
                dd, rows[b0:b1], nz)
            X[off + b0:off + b1, nz + dd["fz"].shape[1]:] = x2_pool[i][rows[b0:b1]]
        ytr[off:off + len(rows)] = y_pool[i][rows]
        off += len(rows)
    assert off == total
    print(f"  matrix {X.shape} = {X.nbytes/2**30:.2f} GiB", flush=True)
    model.fit(X, ytr)
    del X, ytr
    print(f"  fit {time.time()-t:.0f}s iters={model.n_iter_}", flush=True)
    sv2 = S2.score(d, va_idx, xb[va_idx], nzf, model)
    yv = y[va_idx]
    print(f"  held-out AUC={roc_auc_score(yv, sv2):.5f} "
          f"AP={average_precision_score(yv, sv2):.5f} "
          f"(v1 AP={average_precision_score(yv, sv1[0][va_idx]):.5f})", flush=True)

    ntrue = np.bincount(l[va_idx], weights=yv, minlength=n_all).astype(np.int64)
    (thr, macro, extra), curve = M6.sweep(l[va_idx], yv.astype(np.float64), sv2,
                                          n_all, ntrue)
    print(f"  base pool only: thr={thr:.6f} macro={macro:.5f} "
          f"pred={extra[0]:,} tp={extra[1]:,}", flush=True)
    with open(os.path.join(WORK, f"model2b_{split}_meta.txt"), "w") as fh:
        fh.write(f"threshold={thr}\nmacro_fbeta={macro}\n"
                 f"val_ap={average_precision_score(yv, sv2)}\n"
                 f"iters={model.n_iter_}\nn_x2={S2.N_X2}\n")
    with open(os.path.join(WORK, f"model2b_{split}.pkl"), "wb") as fh:
        cols = list(np.load(os.path.join(WORK, "X_train_cols.npy"), allow_pickle=True))
        pickle.dump({"model": model, "thr": thr, "n_x2": S2.N_X2,
                     "cols": cols + list(S2.X2_NAMES)}, fh)
    print("saved model2b_train.pkl", flush=True)

    print("step 4: v2 scores for every pool", flush=True)
    np.save(os.path.join(WORK, f"pairs_{split}_sv2.npy"),
            S2.score(d, np.arange(len(l)), xb, nzf, model))
    for i, tag in enumerate(TAGS):
        x2t = X2[bounds[i + 1]:bounds[i + 2]]
        s = S2.score(S2.pool_d(split, tag, d), np.arange(len(x2t)), x2t, nzf, model)
        np.save(os.path.join(WORK, f"pairs_{split}{tag}_sv2.npy"), s)
        print(f"  {tag}: {len(s):,} rows", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
