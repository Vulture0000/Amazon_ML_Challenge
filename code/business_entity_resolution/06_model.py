"""Stage-2 model: train on kept pairs, tune the threshold for macro F0.5, report metrics."""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
F = importlib.import_module("05_fuzzy")
import common
from common import WORK, Right, Source

VALID_FRAC = float(os.environ.get("VALID_FRAC", "0.1"))
SEED = 7
NEG_SAMPLE = float(os.environ.get("NEG_SAMPLE", "1.0"))


def load(split, keys=("l", "r", "fam", "y", "f", "fz", "sc"), mmap=True):
    d = {}
    for k in keys:
        d[k] = np.load(os.path.join(WORK, f"pairs_{split}_{k}.npy"),
                       mmap_mode="r" if mmap else None)
    return d


def gather(d, idx, n_f, block=2_000_000):
    """Assemble [f | fz] rows for idx without materializing the full matrix."""
    idx = np.asarray(idx)
    nf = n_f
    out = np.empty((len(idx), nf + d["fz"].shape[1]), np.float32)
    for b0 in range(0, len(idx), block):
        b1 = min(b0 + block, len(idx))
        sel = idx[b0:b1]
        out[b0:b1, :nf] = d["f"][sel]
        out[b0:b1, nf:] = d["fz"][sel]
    return out


def sweep(l, y, s, n_left, ntrue):
    """Return (best_thr, best_macro, detail, curve) over a score-quantile grid.

    l/y/s are aligned to the validation split. Thresholds are visited from high
    to low so the selected pair set only grows and the per-entity counts can be
    accumulated incrementally.
    """
    order = np.argsort(-s, kind="stable")
    l, y, s = l[order], y[order], s[order]
    qs = np.unique(np.concatenate([
        np.linspace(0.9990, 0.99999, 60),
        1.0 - np.geomspace(1e-5, 5e-2, 90),
        np.linspace(0.30, 0.90, 40), np.linspace(0.90, 0.9999, 40)]))
    qs = np.unique(np.round(qs, 8))
    best = (None, -1.0, None)
    curve = []
    pos = 0
    npred = np.zeros(n_left, np.int64)
    tp = np.zeros(n_left, np.float64)
    for thr in qs[::-1]:
        k = int(np.searchsorted(-s, -thr, side="right"))
        if k > len(s):
            k = len(s)
        if k <= pos:
            continue
        sel = slice(pos, k)
        npred += np.bincount(l[sel], minlength=n_left)
        tp += np.bincount(l[sel], weights=y[sel], minlength=n_left)
        pos = k
        m = float(common.entity_fbeta(tp, npred, ntrue).mean())
        curve.append((float(thr), m, int(npred.sum()), int(round(tp.sum()))))
        if m > best[1]:
            best = (float(thr), m, (int(npred.sum()), int(round(tp.sum()))))
    return best, curve


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    d = load(split)
    L = Source(f"{split}_s1")
    n_left = len(L)
    n_feat = d["f"].shape[1] + d["fz"].shape[1]
    cols = (F.FUZZ_NAMES + [f"f_{i}" for i in range(d["f"].shape[1])])
    y = np.asarray(d["y"]).astype(np.int8)
    l = np.asarray(d["l"]).astype(np.int64)
    r = np.asarray(d["r"]).astype(np.int64)
    print(f"pairs={len(y):,}  features={n_feat}  positives={int(y.sum()):,} "
          f"rate={y.mean():.6f}", flush=True)
    np.save(os.path.join(WORK, f"X_{split}_cols.npy"), np.array(cols, dtype=object),
            allow_pickle=True)

    rng = np.random.default_rng(SEED)
    n_cover = int(l.max()) + 1
    limited = n_cover < 0.9 * n_left
    if limited:
        print(f"  WARNING: pairs cover only left rows 0..{n_cover-1:,} of {n_left:,} "
              f"(limited run) - metrics are over that prefix only", flush=True)
        n_left = n_cover
    is_valid = np.zeros(n_left, bool)
    nv = int(n_left * VALID_FRAC)
    is_valid[rng.choice(n_left, nv, replace=False)] = True
    vm = is_valid[l]
    print(f"valid pairs={int(vm.sum()):,}  valid entities={nv:,}", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    t = time.time()
    tr_idx = np.flatnonzero(~vm)
    va_idx = np.flatnonzero(vm)
    if NEG_SAMPLE < 1.0:
        neg = np.flatnonzero(y[tr_idx] == 0)
        pos = np.flatnonzero(y[tr_idx] == 1)
        keep_neg = rng.choice(neg, int(len(neg) * NEG_SAMPLE), replace=False)
        tr_idx = np.concatenate([pos, keep_neg])
    model = HistGradientBoostingClassifier(
        max_iter=int(os.environ.get("M2_ITER", "600")), learning_rate=0.06,
        max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
        random_state=0)
    Xtr = gather(d, tr_idx, d["f"].shape[1])
    print(f"  train matrix {Xtr.shape} = {Xtr.nbytes/2**30:.2f} GiB", flush=True)
    model.fit(Xtr, y[tr_idx])
    del Xtr
    sv = np.empty(len(va_idx), np.float32)
    for b0 in range(0, len(va_idx), 2_000_000):
        b1 = min(b0 + 2_000_000, len(va_idx))
        sv[b0:b1] = model.predict_proba(gather(d, va_idx[b0:b1],
                                               d["f"].shape[1]))[:, 1]
    yv = y[va_idx]
    print(f"fit {time.time()-t:.0f}s iters={model.n_iter_}  "
          f"val AUC={roc_auc_score(yv, sv):.5f} AP={average_precision_score(yv, sv):.5f}",
          flush=True)

    ntrue = np.bincount(l[va_idx], weights=yv, minlength=n_left).astype(np.int64)
    (thr, macro, extra), curve = sweep(l[va_idx], yv.astype(np.float64), sv, n_left, ntrue)
    print(f"best threshold={thr:.6f}  macro_f0.5={macro:.5f}  "
          f"pred={extra[0]:,} tp={extra[1]:,}", flush=True)
    for c in curve[len(curve) // 2::max(len(curve) // 12, 1)][:14]:
        print(f"   thr={c[0]:.6f} macro={c[1]:.5f} pred={c[2]:,} tp={c[3]:,}", flush=True)

    pl = l[va_idx][sv >= thr]
    pr = r[va_idx][sv >= thr]
    tl, tr = common.load_gt("train")
    m = is_valid[tl]
    tl, tr = tl[m], tr[m]
    remap = np.full(n_left, -1, np.int64)
    remap[np.flatnonzero(is_valid)] = np.arange(int(is_valid.sum()))
    print(common.fmt("VALID", common.score(remap[pl], pr, remap[tl], tr,
                                            int(is_valid.sum()), Right(split).n)),
          flush=True)
    tl, tr = common.load_gt("train")
    m = tl < n_left
    tl, tr = tl[m], tr[m]
    ntrue_all = np.bincount(tl, minlength=n_left).astype(np.int64)
    print(f"true pairs for {n_left:,} entities = {ntrue_all.sum():,}", flush=True)
    ceil = common.score(l[y == 1], r[y == 1], tl, tr, n_left, Right("train").n)
    print(common.fmt("ALL-POS-AS-PRED (not a bound: F0.5 rewards precision)", ceil), flush=True)
    print(f"blocking recall (kept positives / all true) = "
          f"{y.sum() / max(ntrue_all.sum(), 1):.4f}", flush=True)

    with open(os.path.join(WORK, f"model2_{split}.pkl"), "wb") as fh:
        pickle.dump({"model": model, "thr": thr, "cols": cols}, fh)
    with open(os.path.join(WORK, f"model2_{split}_meta.txt"), "w") as fh:
        fh.write(f"threshold={thr}\nmacro_fbeta={macro}\nval_auc={roc_auc_score(yv, sv)}\n"
                 f"iters={model.n_iter_}\nvalid_frac={VALID_FRAC}\nseed={SEED}\n")
    np.save(os.path.join(WORK, f"pairs_{split}_sv.npy"), sv.astype(np.float32))
    print("saved model2_%s.pkl" % split, flush=True)


if __name__ == "__main__":
    main()
