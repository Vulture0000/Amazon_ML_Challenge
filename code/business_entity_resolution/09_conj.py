"""Round-2 recovery: a *conjunctive* global pass (share a name key AND an address key).

The disjunctive nsuf|askel pass of 08_recover.py is too expensive to run over
every entity (~42B raw candidates), but its coverage of the missed pairs comes
almost entirely from the cases where only one of the two keys is shared.  The
conjunction needs both, which makes the index enormously more selective:

    coverage of 20k missed pairs:  nsuf 79.6%  askel 85.8%  union 98.6%
                                    conjunction 66.9%
    candidates per left row:       0.8   (exact 50-bit cross keys, cap 20,000)

so a *global* pass costs ~1.4M candidates for the whole test set instead of 42
billion.  Cross keys are packed as (nsuf_id << 25) | askel_id and kept exact --
hashing them to 25 bits to match the other families causes so many collisions
that the candidate count explodes 100x (see exp_conj.py).

Rows share a cross key iff they share at least one name suffix-4 *and* at least
one address skeleton token, which is exactly the conjunction measured above.

Usage:
  python 09_conj.py --split train --mask ../../work/mask_train_allvalid.npy --tag _conj
"""
import argparse
import importlib
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
B = importlib.import_module("02_blocking")
R8 = importlib.import_module("08_recover")
R4 = importlib.import_module("04_rank")
import features as FT
from common import PREP, WORK, Right, Source, label_pairs

def left_csr(split, name):
    off, keys = R8.load_csr(f"{split}_s1", name)
    return np.asarray(off), np.asarray(keys)


LEFT_CHUNK = int(os.environ.get("LEFT_CHUNK", "20000"))
PAIR_BATCH = int(os.environ.get("PAIR_BATCH", "4000000"))
KEEP = int(os.environ.get("CONJ_KEEP", "20"))
FAM_BIT = {"naxs": 3}  # reuse the name_pre family bit for the conjunction


def build_index64_full(keys, off):
    """(sorted keys, posting row ids, unique keys, df) for exact 64-bit keys.

    Unlike build_index(), which packs (key << 24 | row) into one array, exact
    64-bit keys cannot be packed that way, so the row of each posting is carried
    through a separate array built from the CSR offsets.
    """
    row_of_key = np.repeat(np.arange(len(off) - 1, dtype=np.int64),
                           np.diff(off).astype(np.int64))
    order = np.argsort(keys, kind="stable")
    inv_keys = keys[order]
    rows = row_of_key[order]
    del keys, order, row_of_key
    starts = np.flatnonzero(np.concatenate(([True], inv_keys[1:] != inv_keys[:-1])))
    kv = inv_keys[starts]
    kc = np.concatenate((starts[1:], [len(inv_keys)])) - starts
    return inv_keys, rows, kv, kc


def cross_keys(noff, nkey, aoff, akey):
    """Per-row cross product of two key sets, as one exact int64 key."""
    nn = np.diff(noff).astype(np.int64)
    na = np.diff(aoff).astype(np.int64)
    cnt = nn * na
    off = np.zeros(len(nn) + 1, np.int64)
    np.cumsum(cnt, out=off[1:])
    total = int(off[-1])
    flat = np.empty(total, np.int64)
    n_st, a_st = noff[:-1].astype(np.int64), aoff[:-1].astype(np.int64)
    n_en, a_en = noff[1:].astype(np.int64), aoff[1:].astype(np.int64)
    w = 0
    for r in range(len(nn)):
        m, k = n_en[r] - n_st[r], a_en[r] - a_st[r]
        if m == 0 or k == 0:
            continue
        blk = (nkey[n_st[r]:n_en[r]].astype(np.int64)[:, None] << 25) | \
            akey[a_st[r]:a_en[r]].astype(np.int64)[None, :]
        flat[w:w + m * k] = blk.ravel()
        w += m * k
    return off, flat, w


def ragged_range_local(starts, lens):
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    out = np.ones(total, np.int64)
    out[0] = starts[0]
    if len(starts) > 1:
        e = np.cumsum(lens)[:-1]
        out[e] = starts[1:] - (starts[:-1] + lens[:-1]) + 1
    return np.cumsum(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--tag", default="_conj")
    ap.add_argument("--cap", type=int, default=20000)
    args = ap.parse_args()
    split, tag = args.split, args.tag
    t_all = time.time()
    L = Source(f"{split}_s1")
    R = Right(split)
    mask = np.load(args.mask)
    sel = np.flatnonzero(mask)
    print(f"{split}: {len(sel):,} of {len(L):,} left rows selected, cap={args.cap}",
          flush=True)

    t = time.time()
    noff, nkey = R8.right_csr(split, "nsuf")
    aoff, akey = R8.right_csr(split, "askel")
    r_off, r_key, used = cross_keys(noff, nkey, aoff, akey)
    print(f"  right cross keys {len(r_key):,} ({len(r_key)/R.n:.1f}/row) "
          f"{time.time()-t:.0f}s", flush=True)
    t = time.time()
    inv_keys, inv_rows, kv, kc = build_index64_full(r_key, r_off)
    print(f"  index built {time.time()-t:.0f}s  distinct keys={len(kv):,} "
          f"max df={int(kc.max()):,}", flush=True)
    del r_key, noff, nkey, aoff, akey

    t = time.time()
    mats = {"n": (FT.tfidf_matrix(L.ntok, L.ntok_off, len(L)),
                  FT.tfidf_matrix(R.ntok, R.ntok_off, R.n)),
            "a": (FT.tfidf_matrix(L.atok, L.atok_off, len(L)),
                  FT.tfidf_matrix(R.atok, R.atok_off, R.n))}
    print(f"  tfidf matrices {time.time()-t:.1f}s", flush=True)
    with open(os.path.join(WORK, "stage1_model.pkl"), "rb") as f:
        stage1 = pickle.load(f)
    n_feat = len(FT.FEATURE_ORDER) + 6
    assert stage1.n_features_in_ == n_feat

    is_train = split == "train"
    buf = {k: [] for k in ("l", "r", "fam", "y", "f", "sc")}
    n_pairs = n_pos = 0
    t0 = time.time()
    lo_all, lk_all = left_csr(split, "nsuf"), left_csr(split, "askel")
    for c0 in range(0, len(sel), LEFT_CHUNK):
        _t = time.time()
        rows = sel[c0:c0 + LEFT_CHUNK]
        keep = np.zeros(len(L), bool)
        keep[rows] = True
        ln_off, ln_key = B.filter_csr(lo_all[0], lo_all[1], keep)
        la_off, la_key = B.filter_csr(lk_all[0], lk_all[1], keep)
        x_off, x_key, _ = cross_keys(ln_off, ln_key, la_off, la_key)
        res = B.expand(x_off, x_key, inv_keys, inv_rows, kv, kc, args.cap,
                       1, left_rows=None, max_out=200_000_000)
        if res is None:
            print(f"  [{c0+len(rows):>8,}] no candidates", flush=True)
            continue
        cl_local, cr = res
        cl = rows[cl_local]
        codes = cl.astype(np.int64) << 27 | cr.astype(np.int64) << 3 | FAM_BIT["naxs"]
        cl, cr, fam = R4.dedup(codes, None)
        n_pairs += len(cl)
        keep_idx = np.empty(len(cl), np.int32)
        chunk_feats = np.empty((len(cl), n_feat), np.float32)
        scores = np.empty(len(cl), np.float32)
        for b0 in range(0, len(cl), PAIR_BATCH):
            b1 = min(b0 + PAIR_BATCH, len(cl))
            out, _ = FT.build_features(L, R, cl[b0:b1], cr[b0:b1], mats=mats)
            Fm = np.empty((b1 - b0, n_feat), np.float32)
            for j, k in enumerate(FT.FEATURE_ORDER):
                Fm[:, j] = out[k]
            for j in range(6):
                Fm[:, len(FT.FEATURE_ORDER) + j] = ((fam[b0:b1] >> j) & 1).astype(np.float32)
            chunk_feats[b0:b1] = Fm
            scores[b0:b1] = stage1.predict_proba(Fm)[:, 1]
        if is_train:
            y = label_pairs(cl, cr).astype(np.float32)
            n_pos += int(y.sum())
        else:
            y = np.zeros(len(cl), np.float32)
        kp = R4.topk_mask(cl, scores, KEEP)
        buf["l"].append(cl[kp].astype(np.int32))
        buf["r"].append(cr[kp].astype(np.int32))
        buf["fam"].append(fam[kp].astype(np.uint8))
        buf["y"].append(y[kp])
        buf["f"].append(chunk_feats[kp])
        buf["sc"].append(scores[kp])
        el = time.time() - t0
        done = c0 + len(rows)
        print(f"  [{done:>8,}/{len(sel):,}] raw={n_pairs:,} "
              f"({n_pairs/done:.1f}/row) kept={sum(len(x) for x in buf['l']):,} "
              f"pos={n_pos:,} {el:.0f}s eta={el/done*(len(sel)-done):.0f}s", flush=True)

    out = {k: np.concatenate(v) for k, v in buf.items() if len(v)}
    for k in ("l", "r", "fam", "y", "sc", "f"):
        out[k] = out[k].astype(np.float32) if k in ("y", "sc") else out[k]
        np.save(os.path.join(WORK, f"pairs_{split}{tag}_{k}.npy"), out[k])
    print(f"kept pairs={len(out['l']):,} ({len(out['l'])/max(len(sel),1):.2f}/selected) "
          f"positives={int(out['y'].sum()):,} total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
