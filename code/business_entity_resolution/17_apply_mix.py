"""Merge any number of candidate pools into the submission, one model each.

The pools disagree about who should score them, which is the whole point of
having a competition ranker: on the validation split the base pool is best
ranked by stage-2b (+0.0108 macro) while the two recall pools, by construction
full of candidates the first pass rejected, are still best ranked by stage 2.
So the merge is specified per pool as (tag, score file, threshold) instead of
assuming one model and one threshold everywhere.

  python 17_apply_mix.py --spec "base:_sv2:0.81,rec:_recv_sv1:0.785,conj:_conj_sv1:0.77"

Reported for every merge: how many entities more than one pass would predict
(they must not silently gain a second prediction), duplicate pairs, and the
row counts, because the official metric only sees the TSV.
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
P7 = importlib.import_module("07_predict")
import common
from common import WORK, Right, Source


def load(spec, split, d):
    """(left, right, score, name) for one entry of --spec."""
    tag, sf, thr = spec.split(":")
    if tag in ("base", "_"):
        tag = ""
    p = os.path.join(WORK, f"pairs_{split}{tag}_")
    l = np.asarray(np.load(p + "l.npy")).astype(np.int64)
    r = np.asarray(np.load(p + "r.npy")).astype(np.int64)
    s = np.asarray(np.load(p + sf.lstrip("_") + ".npy"), np.float32)
    assert len(s) == len(l), f"{sf}: {len(s)} scores for {len(l)} pairs"
    name = f"{tag or '_base'}({sf}@{float(thr):.3f})"
    return l, r, s, float(thr), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--spec", required=True,
                    help="comma-separated tag:scorefile:threshold")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-candidates", action="store_true")
    args = ap.parse_args()
    split = args.split
    t0 = time.time()
    L = Source(f"{split}_s1")
    R = Right(split)
    n_left = len(L)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else P7.OUT

    picks, cand, thrs = [], [], []
    for spec in args.spec.split(","):
        l, r, s, thr, name = load(spec, split, None)
        picks.append((l, r, s))
        cand.append((l, r))
        thrs.append(thr)
        print(f"{name:<28s} {len(l):>10,} pairs", flush=True)

    # Deduplicate before thresholding, keeping the earliest pool's copy, exactly
    # as the validation sweep did.  Thresholding first and then unioning would
    # let a pair rescue itself from a second pool: a candidate the base pool
    # scored 0.50 and the conjunctive pool 0.90 would be rejected under the base
    # threshold but accepted under the conjunctive one, which is a different
    # operating point from the one that was measured.
    pl = np.concatenate([p[0] for p in picks]).astype(np.int64)
    pr = np.concatenate([p[1] for p in picks]).astype(np.int64)
    ps = np.concatenate([p[2] for p in picks]).astype(np.float32)
    pid = np.concatenate([np.full(len(p[0]), i, np.int8)
                          for i, p in enumerate(picks)])
    assert R.n <= (1 << 24), "the dedup key would collide"
    _, first = np.unique((pl << 24) | pr, return_index=True)
    dup = len(pl) - len(first)
    pl, pr, ps, pid = pl[first], pr[first], ps[first], pid[first]
    keep = np.zeros(len(pl), bool)
    ents = []
    for i, t in enumerate(thrs):
        k = (pid == i) & (ps >= t)
        ents.append(np.unique(pl[k]))
        print(f"  pool {i}: {int(k.sum()):>9,} pairs over {len(ents[-1]):>9,} "
              f"entities (thr {t:.3f})", flush=True)
        keep |= k
    pl, pr = pl[keep], pr[keep]
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            both = np.intersect1d(ents[i], ents[j])
            if len(both):
                print(f"  note: {len(both):,} entities receive pairs from pools "
                      f"{i} and {j}", flush=True)
    o = np.lexsort((pr, pl))
    pl, pr = pl[o], pr[o]
    n_ent = len(np.unique(pl))
    print(f"merged: {len(pl):,} pairs over {n_ent:,} entities "
          f"({dup:,} duplicates removed, {n_left-n_ent:,} entities empty)",
          flush=True)

    os.makedirs(out_dir, exist_ok=True)
    s1_ids = np.char.add(np.full(n_left, "S1-", dtype="U3"),
                         np.char.mod("%d", np.asarray(L.ids).astype(np.int64)))
    rids = P7.right_ids(R)
    st = np.flatnonzero(np.concatenate(([True], pl[1:] != pl[:-1])))
    en = np.concatenate((st[1:], [len(pl)]))
    groups = np.full(n_left, "", dtype=object)
    for a, b in zip(st, en):
        groups[pl[a]] = ",".join(rids[pr[a:b]])
    P7.write_tsv(os.path.join(out_dir, "matching_results.tsv"), s1_ids, groups)
    print(f"wrote {out_dir}/matching_results.tsv", flush=True)

    if not args.no_candidates:
        cl = np.concatenate([l for l, _ in cand]).astype(np.int64)
        cr = np.concatenate([r for _, r in cand]).astype(np.int64)
        o = np.lexsort((cr, cl))
        cl, cr = cl[o], cr[o]
        d = np.flatnonzero((cl[1:] == cl[:-1]) & (cr[1:] == cr[:-1]))
        if len(d):
            cl, cr = np.delete(cl, d), np.delete(cr, d)
        st = np.flatnonzero(np.concatenate(([True], cl[1:] != cl[:-1])))
        en = np.concatenate((st[1:], [len(cl)]))
        cgroups = np.full(n_left, "", dtype=object)
        for a, b in zip(st, en):
            cgroups[cl[a]] = ",".join(rids[cr[a:b]])
        P7.write_tsv(os.path.join(out_dir, "candidate_pairs.tsv"), s1_ids, cgroups)
        print(f"wrote {out_dir}/candidate_pairs.tsv ({len(cl):,} candidate pairs, "
              f"{len(d):,} duplicates removed)", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
