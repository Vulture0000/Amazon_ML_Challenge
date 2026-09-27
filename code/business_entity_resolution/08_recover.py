"""Second-chance blocking pass for entities the first pass could not match.

Motivation (exp_miss2.py / exp_miss3.py, honest full-truth validation of the
production pipeline):

  * blocking recall of the six live families is 0.8485, so 15.1% of the true
    pairs are never even proposed;
  * 55.8% of those missed pairs belong to entities that the current pipeline
    predicts *nothing* for, i.e. entities that currently score a flat 0.0;
  * a shared name suffix-4 key (nsuf) plus a shared address skeleton token
    (askel) recovers 98.7% of the missed pairs.

So this pass re-blocks only the entities selected by a caller-supplied mask
(entities with no prediction above the stage-2 threshold), using nsuf + askel,
scores the new candidates with the *existing* stage-1 and stage-2 models and
stores them separately.  Because the mask is disjoint from the entities the
first pass predicts, merging the two prediction sets can only add pairs to
previously empty entities -- it cannot change an existing prediction.

The new families are mapped onto the two generic token columns of the feature
matrix (nsuf -> name_pre, askel -> addr_tok) so the loaded models keep their
expected input width; the family bits are the only thing that differs from a
first-pass candidate.

Usage:
  python 08_recover.py --split train --mask work/mask_train.npy [--caps 2000,2000]
  TAG=_rec python 05_fuzzy.py train
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
B = importlib.import_module("02_blocking")
R4 = importlib.import_module("04_rank")
import features as FT
from common import PREP, WORK, Right, Source, label_pairs

LEFT_CHUNK = int(os.environ.get("LEFT_CHUNK", "10000"))
PAIR_BATCH = int(os.environ.get("PAIR_BATCH", "4000000"))
KEEP = int(os.environ.get("REC_KEEP", "20"))
# production family-bit columns: 0 name_set 1 addr_set 2 house_number
#                             3 name_pre 4 addr_tok 5 name_tok
FAM_BIT = {"nsuf": 3, "askel": 4}


def load_csr(tag, name):
    off = np.load(os.path.join(PREP, f"{tag}_{name}.off.npy"), mmap_mode="r")
    keys = np.load(os.path.join(PREP, f"{tag}_{name}.npy"), mmap_mode="r")
    return off, keys


def right_csr(split, name):
    s2, s3 = load_csr(f"{split}_s2", name), load_csr(f"{split}_s3", name)
    off = np.concatenate([s2[0], s2[0][-1] + s3[0][1:]]).astype(np.int32)
    keys = np.concatenate([np.asarray(s2[1]), np.asarray(s3[1])])
    return off, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--mask", required=True, help=".npy bool over all left rows")
    ap.add_argument("--tag", default="_rec")
    ap.add_argument("--caps", default="2000,2000", help="nsuf_cap,askel_cap")
    args = ap.parse_args()
    split, tag = args.split, args.tag
    caps = dict(zip(("nsuf", "askel"), (int(x) for x in args.caps.split(","))))

    t_all = time.time()
    L = Source(f"{split}_s1")
    R = Right(split)
    mask = np.load(args.mask)
    assert len(mask) == len(L), (len(mask), len(L))
    sel = np.flatnonzero(mask)
    print(f"{split}: {len(sel):,} of {len(L):,} left entities selected "
          f"({len(sel)/len(L):.2%}) caps={caps}", flush=True)

    t = time.time()
    mats = {"n": (FT.tfidf_matrix(L.ntok, L.ntok_off, len(L)),
                  FT.tfidf_matrix(R.ntok, R.ntok_off, R.n)),
            "a": (FT.tfidf_matrix(L.atok, L.atok_off, len(L)),
                  FT.tfidf_matrix(R.atok, R.atok_off, R.n))}
    print(f"tfidf matrices {time.time()-t:.1f}s", flush=True)

    idx = {}
    for nm, cap in caps.items():
        t = time.time()
        roff, rkeys = right_csr(split, nm)
        ik, ir = B.build_index(rkeys, roff, R.n)
        kv, kc = B.key_counts(rkeys)
        idx[nm] = (ik, ir, kv, kc, cap)
        print(f"  index {nm:<6s} {time.time()-t:5.1f}s keys={len(kv):,}", flush=True)

    with open(os.path.join(WORK, "stage1_model.pkl"), "rb") as f:
        stage1 = pickle.load(f)
    n_feat = len(FT.FEATURE_ORDER) + 6
    assert stage1.n_features_in_ == n_feat, (stage1.n_features_in_, n_feat)
    print(f"  stage-1 ranker: {n_feat} features, "
          f"{stage1.n_features_in_} expected", flush=True)

    is_train = split == "train"
    buf = {k: [] for k in ("l", "r", "fam", "y", "f", "sc")}
    n_pairs = n_pos = 0
    t0 = time.time()
    for c0 in range(0, len(sel), LEFT_CHUNK):
        _t = time.time()
        rows = sel[c0:c0 + LEFT_CHUNK]
        keep = np.zeros(len(L), bool)
        keep[rows] = True
        codes = []
        for nm, cap in caps.items():
            loff, lkeys = load_csr(f"{split}_s1", nm)
            loff, lkeys = B.filter_csr(np.asarray(loff), np.asarray(lkeys), keep)
            ik, ir, kv, kc, c = idx[nm]
            res = B.expand(loff, lkeys, ik, ir, kv, kc, c, 1, max_out=60_000_000)
            if res is None:
                continue
            cl, cr = res
            codes.append(cl.astype(np.int64) << 27 | cr.astype(np.int64) << 3
                         | FAM_BIT[nm])
        if not codes:
            continue
        cl, cr, fam = R4.dedup(np.concatenate(codes), None)
        n_pairs += len(cl)
        chunk_feats = np.empty((len(cl), n_feat), np.float32)
        scores = np.empty(len(cl), np.float32)
        for b0 in range(0, len(cl), PAIR_BATCH):
            b1 = min(b0 + PAIR_BATCH, len(cl))
            out, _ = FT.build_features(L, R, rows[cl[b0:b1]], cr[b0:b1], mats=mats)
            F = np.empty((b1 - b0, n_feat), np.float32)
            for j, k in enumerate(FT.FEATURE_ORDER):
                F[:, j] = out[k]
            for j in range(6):
                F[:, len(FT.FEATURE_ORDER) + j] = ((fam[b0:b1] >> j) & 1).astype(np.float32)
            chunk_feats[b0:b1] = F
            scores[b0:b1] = stage1.predict_proba(F)[:, 1]
        if is_train:
            y = label_pairs(rows[cl], cr).astype(np.float32)
            n_pos += int(y.sum())
        else:
            y = np.zeros(len(cl), np.float32)
        kp = R4.topk_mask(rows[cl], scores, KEEP)
        buf["l"].append(rows[cl[kp]].astype(np.int32))
        buf["r"].append(cr[kp].astype(np.int32))
        buf["fam"].append(fam[kp].astype(np.uint8))
        buf["y"].append(y[kp])
        buf["f"].append(chunk_feats[kp])
        buf["sc"].append(scores[kp])
        el = time.time() - t0
        print(f"  [{c0+len(rows):>8,}/{len(sel):,}] raw={n_pairs:,} kept="
              f"{sum(len(x) for x in buf['l']):,} pos={n_pos:,} "
              f"rate={n_pos/max(n_pairs,1):.4f} {el:.0f}s "
              f"eta={el/max(c0+len(rows),1)*(len(sel)-c0-len(rows)):.0f}s", flush=True)

    out = {k: np.concatenate(v) for k, v in buf.items() if len(v)}
    out["l"] = out["l"].astype(np.int32)
    out["r"] = out["r"].astype(np.int32)
    for k in ("l", "r", "fam", "y", "sc", "f"):
        np.save(os.path.join(WORK, f"pairs_{split}{tag}_{k}.npy"), out[k])
    print(f"kept pairs={len(out['l']):,} ({len(out['l'])/max(len(sel),1):.2f}/selected) "
          f"positives={int(out['y'].sum()):,} recall_of_missed="
          f"{out['y'].sum()/max(n_pos,1):.4f} total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
