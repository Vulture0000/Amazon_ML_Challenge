"""Stage 1: blocking + cheap features + learned pruning, chunked by left record.

Writes work/pairs_<split>_{l,r,fam,y,sc}.npy  (top-K per left entity by stage-1 score).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
B = importlib.import_module("02_blocking")
import features as FT
from common import WORK, Right, Source, label_pairs

KEEP = int(os.environ.get("KEEP", "20"))
LEFT_CHUNK = int(os.environ.get("LEFT_CHUNK", "10000"))
PAIR_BATCH = int(os.environ.get("PAIR_BATCH", "6000000"))
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.0022"))
FIT_PER_CHUNK = int(os.environ.get("FIT_PER_CHUNK", "150000"))
SEED = 12345

# (name, cap)  -- order matters: cheap+precise first, so the tail can be skipped
CAPS = {k: int(v) for k, v in (kv.split("=") for kv in
       os.environ.get("CAPS", "name_set=2000,addr_set=100,house_number=2000,"
                              "name_pre=2000,addr_tok=1000,name_tok=2000").split(","))}
FAMILIES = [
    ("name_set", CAPS["name_set"], "set"),
    ("addr_set", CAPS["addr_set"], "set"),
    ("house_number", CAPS["house_number"], "num"),
    ("name_pre", CAPS["name_pre"], "tok"),
    ("addr_tok", CAPS["addr_tok"], "tok"),
    ("name_tok", CAPS["name_tok"], "tok"),
]
FAM_NAMES = [f[0] for f in FAMILIES]


def build_family_inputs(L, R, ntok, atok, npre, rows):
    """Left-side key arrays (chunk-local rows) for every blocking family."""
    RIDX = np.arange(R.n + 1, dtype=np.int32)
    NL = np.arange(len(rows) + 1, dtype=np.int32)
    l_hn = np.asarray(L.hnum)[rows]
    n = len(rows)
    return {
        "name_set": (NL, B.set_key(ntok[1], ntok[0], 0x5DEECE66D).astype(np.int32),
                     RIDX, B.set_key(np.asarray(R.ntok), R.ntok_off, 0x5DEECE66D).astype(np.int32)),
        "addr_set": (NL, B.set_key(atok[1], atok[0], 0x2545F491).astype(np.int32),
                     RIDX, B.set_key(np.asarray(R.atok), R.atok_off, 0x2545F491).astype(np.int32)),
        "house_number": (NL, B.num_key(l_hn, 0x1234567, l_hn > 0).astype(np.int32),
                         RIDX, B.num_key(np.asarray(R.hnum), 0x1234567,
                                         np.asarray(R.hnum) > 0).astype(np.int32)),
        "name_pre": (npre[0], npre[1].astype(np.int32), R.npre_off,
                     np.asarray(R.npre).astype(np.int32)),
        "addr_tok": (atok[0], atok[1].astype(np.int32), R.atok_off,
                     np.asarray(R.atok).astype(np.int32)),
        "name_tok": (ntok[0], ntok[1].astype(np.int32), R.ntok_off,
                     np.asarray(R.ntok).astype(np.int32)),
    }


def dedup(codes, npair_per_fam):
    """codes = left<<27 | right<<3 | fam.  -> (left, right, family bitmask), sorted by left."""
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    pair = codes >> 3
    new = np.empty(len(pair), bool)
    new[0] = True
    np.not_equal(pair[1:], pair[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    fam = np.bitwise_or.reduceat((1 << (codes & 7)).astype(np.uint8), starts)
    return (pair[starts] >> 24), (pair[starts] & ((1 << 24) - 1)), fam


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    t_all = time.time()
    L = Source(f"{split}_s1")
    R = Right(split)
    n_left_all = len(L)
    n_left = n_left_all
    n_limit = int(os.environ.get("N_LIMIT", "0"))
    if n_limit:
        n_left = min(n_left_all, n_limit)
    is_train = split == "train"
    print(f"{split}: n_left={n_left:,} n_right={R.n:,}", flush=True)

    t = time.time()
    mats = {"n": (FT.tfidf_matrix(L.ntok, L.ntok_off, n_left_all),
                  FT.tfidf_matrix(R.ntok, R.ntok_off, R.n)),
            "a": (FT.tfidf_matrix(L.atok, L.atok_off, n_left_all),
                  FT.tfidf_matrix(R.atok, R.atok_off, R.n))}
    print(f"tfidf matrices {time.time()-t:.1f}s", flush=True)

    idx = {}
    for nm, cap, kind in FAMILIES:
        t = time.time()
        if kind == "set":
            rtok = {"name_set": (np.asarray(R.ntok), R.ntok_off, 0x5DEECE66D),
                    "addr_set": (np.asarray(R.atok), R.atok_off, 0x2545F491)}[nm]
            ik, ir = B.build_index(rtok[0], rtok[1], R.n)
            kv, kc = B.key_counts(rtok[0])
        elif kind == "num":
            rh = np.asarray(R.hnum)
            ik, ir = B.build_index(B.num_key(rh, 0x1234567, rh > 0).astype(np.int32),
                                   np.arange(R.n + 1, dtype=np.int32), R.n)
            kv, kc = B.key_counts(B.num_key(rh, 0x1234567, rh > 0).astype(np.int32))
        else:
            rk = {"name_pre": (np.asarray(R.npre), R.npre_off),
                  "addr_tok": (np.asarray(R.atok), R.atok_off),
                  "name_tok": (np.asarray(R.ntok), R.ntok_off)}[nm]
            ik, ir = B.build_index(rk[0], rk[1], R.n)
            kv, kc = B.key_counts(rk[0])
        idx[nm] = (ik, ir, kv, kc, cap)
        print(f"  index {nm:<12s} {time.time()-t:5.1f}s  keys={len(kv):,}", flush=True)

    n_fit_chunks = max(1, int(round(TRAIN_FRAC * n_left / max(LEFT_CHUNK, 1))))
    stage1 = None
    if (not is_train) or os.environ.get("STAGE1_LOAD", "0") != "0":
        import pickle
        with open(os.path.join(WORK, "stage1_model.pkl"), "rb") as f:
            stage1 = pickle.load(f)
        print("  loaded stage-1 ranker from work/stage1_model.pkl", flush=True)
    else:
        print(f"  stage-1 fit from first {n_fit_chunks} chunks, "
              f"{FIT_PER_CHUNK:,} rows/chunk", flush=True)
    fit_feats, fit_y = [], []
    buffers = {k: [] for k in ("l", "r", "fam", "y", "f", "sc")}
    n_seen = 0
    n_pairs = 0
    n_pos = 0
    t0 = time.time()
    TM = dict.fromkeys(("expand", "dedup", "feat", "label", "predict", "topk", "io"), 0.0)
    for start in range(0, n_left, LEFT_CHUNK):
        _t = time.time()
        rows = np.arange(start, min(start + LEFT_CHUNK, n_left))
        keep = np.zeros(n_left_all, bool)
        keep[rows] = True
        ntok = B.filter_csr(L.ntok_off, np.asarray(L.ntok), keep)
        atok = B.filter_csr(L.atok_off, np.asarray(L.atok), keep)
        npre = B.filter_csr(L.npre_off, np.asarray(L.npre), keep)
        fams = build_family_inputs(L, R, ntok, atok, npre, rows)
        codes = []
        for fi, (nm, cap, kind) in enumerate(FAMILIES):
            ik, ir, kv, kc, c = idx[nm]
            f = fams[nm]
            res = B.expand(f[0], f[1], ik, ir, kv, kc, c, 1, max_out=120_000_000)
            if res is None:
                continue
            cl, cr = res
            codes.append(cl.astype(np.int64) << 27 | cr.astype(np.int64) << 3 | fi)
        if not codes:
            continue
        codes = np.concatenate(codes)
        del fams
        TM["expand"] += time.time() - _t
        _t = time.time()
        cl, cr, fam = dedup(codes, None)
        del codes
        TM["dedup"] += time.time() - _t
        _t = time.time()
        n_pairs += len(cl)
        NF = len(FT.FEATURE_ORDER) + len(FAM_NAMES)
        chunk_feats = np.empty((len(cl), NF), np.float32)
        scores = np.empty(len(cl), np.float32)
        for b0 in range(0, len(cl), PAIR_BATCH):
            b1 = min(b0 + PAIR_BATCH, len(cl))
            out, _ = FT.build_features(L, R, rows[cl[b0:b1]], cr[b0:b1], mats=mats)
            F = np.empty((b1 - b0, NF), np.float32)
            for j, k in enumerate(FT.FEATURE_ORDER):
                F[:, j] = out[k]
            for j, nm in enumerate(FAM_NAMES):
                F[:, len(FT.FEATURE_ORDER) + j] = ((fam[b0:b1] >> j) & 1).astype(np.float32)
            chunk_feats[b0:b1] = F
            del out
            if stage1 is not None:
                _p = time.time()
                scores[b0:b1] = stage1.predict_proba(F)[:, 1]
                TM["predict"] += time.time() - _p
        TM["feat"] += time.time() - _t
        _t = time.time()
        if is_train:
            y = label_pairs(rows[cl], cr).astype(np.float32)
            n_pos += int(y.sum())
        else:
            y = np.zeros(len(cl), np.float32)
        TM["label"] += time.time() - _t
        _t = time.time()
        if stage1 is None:
            rs2 = np.random.default_rng(1000 + start)
            take = min(len(y), FIT_PER_CHUNK)
            sub2 = rs2.choice(len(y), take, replace=False)
            fit_feats.append(chunk_feats[sub2])
            fit_y.append(y[sub2])
        else:
            keep = topk_mask(rows[cl], scores, KEEP)
            buffers["l"].append(rows[cl[keep]].astype(np.int32))
            buffers["r"].append(cr[keep].astype(np.int32))
            buffers["fam"].append(fam[keep].astype(np.uint8))
            buffers["y"].append(y[keep])
            buffers["f"].append(chunk_feats[keep])
            buffers["sc"].append(scores[keep])
        TM["topk"] += time.time() - _t
        n_seen += len(rows)
        el = time.time() - t0
        if int(os.environ.get("PROFILE", "0")):
            print("      timings " + " ".join(f"{k}={v:.1f}" for k, v in TM.items()), flush=True)
        print(f"  [{n_seen/n_left:6.1%}] {n_seen:,}/{n_left:,} pairs={n_pairs:,} "
              f"pos={n_pos:,} rate={n_pos/max(n_pairs,1):.4f} {el:.0f}s "
              f"eta={el/max(n_seen,1)*(n_left-n_seen):.0f}s", flush=True)
        if stage1 is None and (start // LEFT_CHUNK) + 1 >= n_fit_chunks:
            print("  training stage-1 ranker ...", flush=True)
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import roc_auc_score
            Xf = np.concatenate(fit_feats)
            yf = np.concatenate(fit_y)
            rs = np.random.default_rng(0)
            nsub = min(len(yf), 3_500_000)
            sub = rs.choice(len(yf), nsub, replace=False)
            stage1 = HistGradientBoostingClassifier(
                max_iter=int(os.environ.get("S1_ITER", "120")), learning_rate=0.15,
                max_leaf_nodes=31, min_samples_leaf=2000, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                random_state=0)
            vs = np.zeros(len(yf), bool)
            vs[rs.choice(len(yf), int(len(yf) * 0.15), replace=False)] = True
            tr_m = sub & ~vs
            stage1.fit(Xf[tr_m], yf[tr_m])
            print(f"  stage-1 val AUC = {roc_auc_score(yf[vs], stage1.predict_proba(Xf[vs])[:,1]):.4f} "
                  f"iters={stage1.n_iter_} fit_rows={int(tr_m.sum()):,}", flush=True)
            keep = topk_mask(rows[cl], stage1.predict_proba(chunk_feats)[:, 1], KEEP)
            buffers["l"].append(rows[cl[keep]].astype(np.int32))
            buffers["r"].append(cr[keep].astype(np.int32))
            buffers["fam"].append(fam[keep].astype(np.uint8))
            buffers["y"].append(y[keep])
            buffers["f"].append(chunk_feats[keep])
            buffers["sc"].append(stage1.predict_proba(chunk_feats[keep])[:,1].astype(np.float32))
            fit_feats, fit_y = None, None

    out = {k: np.concatenate(v) for k, v in buffers.items() if len(v)}
    out["l"] = out["l"].astype(np.int32)
    out["r"] = out["r"].astype(np.int32)
    np.save(os.path.join(WORK, f"pairs_{split}_l.npy"), out["l"])
    np.save(os.path.join(WORK, f"pairs_{split}_r.npy"), out["r"])
    np.save(os.path.join(WORK, f"pairs_{split}_fam.npy"), out["fam"])
    np.save(os.path.join(WORK, f"pairs_{split}_y.npy"), out["y"])
    np.save(os.path.join(WORK, f"pairs_{split}_f.npy"), out["f"].astype(np.float32))
    np.save(os.path.join(WORK, f"pairs_{split}_sc.npy"), out["sc"])
    print(f"kept pairs={len(out['l']):,} ({len(out['l'])/n_left:.2f}/left) "
          f"positives={int(out['y'].sum()):,} recall={out['y'].sum()/max(n_pos,1):.4f} "
          f"total {time.time()-t_all:.0f}s", flush=True)


def topk_mask(left, score, k):
    """Mask of the top-k scores within each left group (any input order)."""
    n = len(left)
    if n == 0:
        return np.zeros(0, bool)
    order = np.lexsort((-np.asarray(score, np.float64), np.asarray(left, np.int64)))
    ls = np.asarray(left, np.int64)[order]
    new = np.empty(n, bool)
    new[0] = True
    np.not_equal(ls[1:], ls[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    rank = np.arange(n) - np.repeat(starts, np.diff(np.append(starts, n)))
    keep_sorted = rank < k
    keep = np.zeros(n, bool)
    keep[order[keep_sorted]] = True
    return keep


if __name__ == "__main__":
    main()
