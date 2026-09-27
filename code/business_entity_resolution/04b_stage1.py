"""Train the stage-1 pruning ranker on raw candidates and save it for the test pass."""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
B = importlib.import_module("02_blocking")
import features as FT
import common
from common import WORK, Right, Source, label_pairs

M = importlib.import_module("04_rank")

N_SAMPLE = int(os.environ.get("STAGE1_SAMPLE", "20000"))
NEG_PER_CHUNK = int(os.environ.get("NEG_PER_CHUNK", "120000"))
LEFT_CHUNK = 6000


def chunk_candidates(L, R, mats, idx, rows, n_left, NF):
    """Build (chunk_feats, fam, n_pos_raw) for one chunk of left rows."""
    keep = np.zeros(n_left, bool)
    keep[rows] = True
    ntok = B.filter_csr(L.ntok_off, np.asarray(L.ntok), keep)
    atok = B.filter_csr(L.atok_off, np.asarray(L.atok), keep)
    npre = B.filter_csr(L.npre_off, np.asarray(L.npre), keep)
    fams = M.build_family_inputs(L, R, ntok, atok, npre, rows)
    codes = []
    for fi, (nm, cap, kind) in enumerate(M.FAMILIES):
        ik, ir, kv, kc, c = idx[nm]
        f = fams[nm]
        res = B.expand(f[0], f[1], ik, ir, kv, kc, c, 1, max_out=120_000_000)
        if res is not None:
            a, b = res
            codes.append(a.astype(np.int64) << 27 | b.astype(np.int64) << 3 | fi)
    del fams
    cl, cr, fam = M.dedup(np.concatenate(codes), None)
    del codes
    out, _ = FT.build_features(L, R, rows[cl], cr, mats=mats)
    F = np.empty((len(cl), NF), np.float32)
    for j, k in enumerate(FT.FEATURE_ORDER):
        F[:, j] = out[k]
    for j, nm in enumerate(M.FAM_NAMES):
        F[:, len(FT.FEATURE_ORDER) + j] = ((fam >> j) & 1).astype(np.float32)
    del out
    return F, fam, cl, cr


def main():
    L = Source("train_s1")
    R = Right("train")
    n_left = len(L)
    mats = {"n": (FT.tfidf_matrix(L.ntok, L.ntok_off, n_left),
                  FT.tfidf_matrix(R.ntok, R.ntok_off, R.n)),
            "a": (FT.tfidf_matrix(L.atok, L.atok_off, n_left),
                  FT.tfidf_matrix(R.atok, R.atok_off, R.n))}
    idx = {}
    for nm, cap, kind in M.FAMILIES:
        if kind == "set":
            rt = {"name_set": (np.asarray(R.ntok), R.ntok_off),
                  "addr_set": (np.asarray(R.atok), R.atok_off)}[nm]
            ik, ir = B.build_index(rt[0], rt[1], R.n)
            kv, kc = B.key_counts(rt[0])
        elif kind == "num":
            rh = np.asarray(R.hnum)
            rk = B.num_key(rh, 0x1234567, rh > 0).astype(np.int32)
            ik, ir = B.build_index(rk, np.arange(R.n + 1, dtype=np.int32), R.n)
            kv, kc = B.key_counts(rk)
        else:
            rk = {"name_pre": (np.asarray(R.npre), R.npre_off),
                  "addr_tok": (np.asarray(R.atok), R.atok_off),
                  "name_tok": (np.asarray(R.ntok), R.ntok_off)}[nm]
            ik, ir = B.build_index(rk[0], rk[1], R.n)
            kv, kc = B.key_counts(rk[0])
        idx[nm] = (ik, ir, kv, kc, cap)

    rng = np.random.default_rng(M.SEED)
    n_fit_chunks = max(1, N_SAMPLE // LEFT_CHUNK)
    order = rng.permutation(n_left)[:(n_fit_chunks + 2) * LEFT_CHUNK]
    feats, ys = [], []
    NF = len(FT.FEATURE_ORDER) + len(M.FAM_NAMES)
    t0 = time.time()
    for ci, c0 in enumerate(range(0, n_fit_chunks * LEFT_CHUNK, LEFT_CHUNK)):
        rows = np.sort(order[c0:c0 + LEFT_CHUNK])
        chunk_feats, fam, cl, cr = chunk_candidates(L, R, mats, idx, rows,
                                                    n_left, NF)
        y = label_pairs(rows[cl], cr)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        take_neg = min(len(neg), NEG_PER_CHUNK)
        sub = np.concatenate([pos, rng.choice(neg, take_neg, replace=False)])
        feats.append(chunk_feats[sub])
        ys.append(y[sub])
        print(f"  chunk {ci+1}/{n_fit_chunks} rows {rows[-1]:,} cand={len(cl):,} "
              f"pos={len(pos):,} fit={sum(len(f) for f in feats):,} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del chunk_feats, y

    Xf = np.concatenate(feats)
    yf = np.concatenate(ys).astype(np.int32)
    print(f"fit matrix {Xf.shape}  positives={int(yf.sum()):,} "
          f"rate={yf.mean():.5f}", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    vs = np.zeros(len(yf), bool)
    vs[rng.choice(len(yf), int(len(yf) * 0.1), replace=False)] = True
    model = HistGradientBoostingClassifier(
        max_iter=int(os.environ.get("S1_ITER", "300")), learning_rate=0.12,
        max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=0)
    t = time.time()
    model.fit(Xf[~vs], yf[~vs])
    pv = model.predict_proba(Xf[vs])[:, 1]
    print(f"fit {time.time()-t:.0f}s iters={model.n_iter_}  val "
          f"AUC={roc_auc_score(yf[vs], pv):.5f} "
          f"AP={average_precision_score(yf[vs], pv):.5f} "
          f"train_pos={int(yf[~vs].sum()):,}", flush=True)
    del Xf, yf

    # --- held-out pruning check on the chunk right after the fit chunks ---
    for kk in (10, 20, 40):
        c0 = n_fit_chunks * LEFT_CHUNK + (kk // 20) * LEFT_CHUNK
        rows = np.sort(order[c0:c0 + LEFT_CHUNK])
        Fe, fam, cl, cr = chunk_candidates(L, R, mats, idx, rows, n_left, NF)
        ye = label_pairs(rows[cl], cr)
        pr = model.predict_proba(Fe)[:, 1]
        for k in (10, 20, 40):
            sel = M.topk_mask(rows[cl], pr, k)
            print(f"  held-out chunk@{c0}: pruning recall@top{k} = "
                  f"{ye[sel].sum()/max(ye.sum(),1):.4f} "
                  f"({int(ye[sel].sum()):,}/{int(ye.sum()):,}) "
                  f"cand={len(cl):,} kept={int(sel.sum()):,}", flush=True)
        del Fe, ye, pr
        if kk == 10:
            break

    with open(os.path.join(WORK, "stage1_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("saved", os.path.join(WORK, "stage1_model.pkl"), flush=True)


if __name__ == "__main__":
    main()
