"""Stage-2b: a ranker that can see who else wants the same record.

The pairwise features cannot express the one-to-one prior of entity resolution:
"how hard is this entity competing for this right record?"  Two thirds of the
remaining macro loss sits in entities that are matched *partially*, which is
exactly a ranking failure, and the measured effect of the features below on
that bucket is +0.0185 macro (0.92803 -> 0.94655) at equal training budget.

The features are therefore built from the score of the *deployed* stage-2 model
over the whole candidate universe, which makes this a two-stage ranker:

    v1 scores every candidate            (already saved as pairs_*_sv*.npy)
    X2  = competition features over the union of all candidate pools
    v2  scores every candidate on [cheap | fuzzy | X2]

Because the universe is the union of the base pool and every recall pool, the
claim counts and margins are identical at training and inference time.  Nothing
here changes blocking, the fuzzy features or the recovery passes; it only
re-scores the candidates that already exist.
"""
import importlib
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M6 = importlib.import_module("06_model")
from common import WORK, Right

N_X2 = 10
X2_NAMES = ("claim_margin", "claim_rank", "claims", "first_pass",
            "ent_margin", "src_margin", "crowd_r", "crowd_l", "rank1", "is_s2")
CHUNK = 2_000_000


def _rank_by(key, score, n):
    """1-based rank of each row's key when rows are ordered by score desc."""
    o = np.lexsort((-score, key))
    k = key[o]
    start = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
    r = np.empty(len(key), np.float32)
    r[o] = (np.arange(len(key))
            - np.repeat(start, np.diff(np.append(start, len(key))))) + 1.0
    return r


def _max_by(key, score, n):
    n = max(n, int(key.max()) + 1 if len(key) else 1)
    out = np.full(n, -np.inf, np.float32)
    np.maximum.at(out, key, score)
    return out


def extras(l, r, sc, sv, R, base_thr, n2=None):
    """The N_X2 competition/global features for one candidate universe."""
    n = len(l)
    n2 = N_X2 if n2 is None else n2
    out = np.empty((n, n2), np.float32)
    claim_best = _max_by(r, sv, R.n)
    claim_rank = _rank_by(r, sv, R.n)
    out[:, 0] = sv - claim_best[r]
    out[:, 1] = np.log1p(claim_rank)
    out[:, 2] = np.log1p(np.bincount(r, minlength=R.n).astype(np.float32)[r])
    out[:, 3] = (sv >= base_thr).astype(np.float32)
    src = (r >= R.n2).astype(np.int64)
    key2 = l * 2 + src
    out[:, 4] = sv - _max_by(key2, sv, 0)[key2]
    # margin over the entity's own runner-up, i.e. how dominant the claim is
    o = np.lexsort((-sv, l))
    lo = l[o]
    st = np.flatnonzero(np.concatenate(([True], lo[1:] != lo[:-1])))
    grp = np.repeat(st, np.diff(np.append(st, n)))
    cnt = np.bincount(grp)
    nxt = np.minimum(grp + 1, n - 1)
    runner = np.where(cnt[grp] >= 2, sv[o][nxt], sv[o])
    inv = np.empty(n, np.int64)
    inv[o] = np.arange(n)
    out[:, 5] = (runner - sv[o])[inv]
    if n2 >= 7:
        out[:, 6] = np.log1p(np.bincount(r, minlength=R.n).astype(np.float32)[r])
        out[:, 7] = np.log1p(np.bincount(l).astype(np.float32)[l])
        out[:, 8] = np.log1p(_rank_by(l, sc, 0))
        out[:, 9] = (r < R.n2).astype(np.float32)
    return out


def assemble(d, rows, x2, nzf, n_x2=N_X2, chunk=CHUNK):
    """[cheap f | fuzzy fz | X2] for `rows`, built chunk by chunk."""
    nfz = d["fz"].shape[1]
    X = np.empty((len(rows), nzf + nfz + n_x2), np.float32)
    for b0 in range(0, len(rows), chunk):
        b1 = min(b0 + chunk, len(rows))
        X[b0:b1, :nzf + nfz] = M6.gather(d, rows[b0:b1], nzf)
        X[b0:b1, nzf + nfz:] = x2[b0:b1, :n_x2]
    return X


def score(d, rows, x2, nzf, model, n_x2=N_X2, chunk=CHUNK):
    out = np.empty(len(rows), np.float32)
    for b0 in range(0, len(rows), chunk):
        b1 = min(b0 + chunk, len(rows))
        out[b0:b1] = model.predict_proba(
            assemble(d, rows[b0:b1], x2[b0:b1], nzf, n_x2))[:, 1]
    return out


def pool(split, tag):
    """(l, r, y, sc, f, fz, sv1) for one pool; fz is None for the base pool when
    it is already on disk under its own name."""
    p = os.path.join(WORK, f"pairs_{split}{tag}_")
    l = np.load(p + "l.npy").astype(np.int64)
    r = np.load(p + "r.npy").astype(np.int64)
    y = np.load(p + "y.npy") if os.path.exists(p + "y.npy") else None
    sc = np.load(p + "sc.npy").astype(np.float32)
    sv = np.load(p + "sv1.npy") if os.path.exists(p + "sv1.npy") else None
    return l, r, y, sc, sv


def load_v1(split, tag, d, nzf, model):
    """v1 score for a pool, computing and caching it if needed."""
    p = os.path.join(WORK, f"pairs_{split}{tag}_sv1.npy")
    if os.path.exists(p):
        return np.load(p)
    l = np.load(os.path.join(WORK, f"pairs_{split}{tag}_l.npy"))
    n = len(l)
    sub = pool_d(split, tag, d)
    out = np.empty(n, np.float32)
    for b0 in range(0, n, CHUNK):
        b1 = min(b0 + CHUNK, n)
        out[b0:b1] = model.predict_proba(
            M6.gather(sub, np.arange(b0, b1), nzf))[:, 1]
    np.save(p, out)
    return out


def pool_d(split, tag, base_d=None):
    """The feature matrices belonging to one pool.

    A pool's rows index *its own* f/fz, never the base pool's: 05_fuzzy writes
    per-tag arrays, and gather() indexes them by row, so passing the base
    matrices scores a recall pool against the wrong features entirely.
    """
    if tag == "":
        return base_d
    p = os.path.join(WORK, f"pairs_{split}{tag}_")
    return {"f": np.load(p + "f.npy", mmap_mode="r"),
            "fz": np.load(p + "fz.npy", mmap_mode="r")}


def load_v2(split, tag, d, nzf, model2, n_x2=N_X2):
    """v2 score for a pool, computing and caching it if needed."""
    p = os.path.join(WORK, f"pairs_{split}{tag}_sv2.npy")
    if os.path.exists(p):
        return np.load(p)
    x2 = np.load(os.path.join(WORK, f"pairs_{split}{tag}_x2.npy"), mmap_mode="r")
    n = len(x2)
    out = np.empty(n, np.float32)
    for b0 in range(0, n, CHUNK):
        b1 = min(b0 + CHUNK, n)
        out[b0:b1] = model2.predict_proba(
            assemble(d, np.arange(b0, b1), x2[b0:b1], nzf, n_x2))[:, 1]
    np.save(p, out)
    return out
