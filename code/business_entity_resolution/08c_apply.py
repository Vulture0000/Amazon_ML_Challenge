"""Merge the recovery pass into the final submission.

The recovery pass only produced candidates for entities the first pass left
empty, so merging is additive: for those entities the first pass contributes no
pair at all.  Pairs are still de-duplicated defensively, and the two threshold
values (base_thr from the model, rec_thr from the validation sweep) are applied
to their own score columns.

Usage:
  python 08c_apply.py --split test --tag _recv --rec-thr 0.794601 --out-dir ../output
"""
import argparse
import os
import pickle
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
M6 = importlib.import_module("06_model")
P7 = importlib.import_module("07_predict")
import common
from common import WORK, Right, Source


def score_with_model2(split, tag, path_l, path_f, path_fz):
    with open(os.path.join(WORK, "model2_train.pkl"), "rb") as f:
        mdl = pickle.load(f)
    d = {"l": np.load(path_l, mmap_mode="r"),
         "f": np.load(path_f, mmap_mode="r"),
         "fz": np.load(path_fz, mmap_mode="r")}
    nf = d["f"].shape[1]
    s = np.empty(len(d["l"]), np.float32)
    for b0 in range(0, len(d["l"]), 2_000_000):
        b1 = min(b0 + 2_000_000, len(d["l"]))
        s[b0:b1] = mdl["model"].predict_proba(M6.gather(d, np.arange(b0, b1), nf))[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="_recv")
    ap.add_argument("--rec-thr", type=float, required=True)
    ap.add_argument("--out-dir", default=None, help="default: write next to the current output")
    ap.add_argument("--backup", action="store_true", help="copy the current TSVs first")
    ap.add_argument("--rescore", action="store_true", help="ignore a cached recovery score")
    ap.add_argument("--no-candidates", action="store_true",
                    help="leave candidate_pairs.tsv untouched")
    args = ap.parse_args()
    split, tag = args.split, args.tag
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else P7.OUT

    L = Source(f"{split}_s1")
    R = Right(split)
    n_left = len(L)
    with open(os.path.join(WORK, "model2_train.pkl"), "rb") as f:
        base_thr = float(pickle.load(f)["thr"])

    bl = np.load(os.path.join(WORK, f"pairs_{split}_l.npy"), mmap_mode="r")
    br = np.load(os.path.join(WORK, f"pairs_{split}_r.npy"), mmap_mode="r")
    bs = np.load(os.path.join(WORK, f"pairs_{split}_sv.npy"), mmap_mode="r")
    print(f"baseline: {len(bl):,} candidate pairs, thr={base_thr:.6f}", flush=True)

    rl = np.load(os.path.join(WORK, f"pairs_{split}{tag}_l.npy"), mmap_mode="r")
    rr = np.load(os.path.join(WORK, f"pairs_{split}{tag}_r.npy"), mmap_mode="r")
    sv_path = os.path.join(WORK, f"pairs_{split}{tag}_sv.npy")
    if os.path.exists(sv_path) and not args.rescore:
        rs = np.load(sv_path)
        print(f"reusing {sv_path}", flush=True)
    else:
        print(f"scoring {len(rl):,} recovery pairs ...", flush=True)
        t0 = time.time()
        rs = score_with_model2(split, tag,
                               os.path.join(WORK, f"pairs_{split}{tag}_l.npy"),
                               os.path.join(WORK, f"pairs_{split}{tag}_f.npy"),
                               os.path.join(WORK, f"pairs_{split}{tag}_fz.npy"))
        print(f"  {time.time()-t0:.0f}s", flush=True)
        np.save(sv_path, rs)

    keep_b = np.asarray(bs) >= base_thr
    keep_r = rs >= args.rec_thr
    print(f"baseline predicted pairs : {int(keep_b.sum()):,} "
          f"({len(np.unique(np.asarray(bl)[keep_b])):,} entities)", flush=True)
    print(f"recovery predicted pairs : {int(keep_r.sum()):,} "
          f"({len(np.unique(np.asarray(rl)[keep_r])):,} entities, thr={args.rec_thr:.6f})",
          flush=True)
    # the two sets must be entity-disjoint, otherwise the merge would silently
    # add a second prediction to an entity that already has one
    both = np.intersect1d(np.unique(np.asarray(bl)[keep_b]),
                         np.unique(np.asarray(rl)[keep_r]))
    print(f"entities predicted by both passes: {len(both):,}", flush=True)

    pl = np.concatenate([np.asarray(bl)[keep_b], np.asarray(rl)[keep_r]]).astype(np.int64)
    pr = np.concatenate([np.asarray(br)[keep_b], np.asarray(rr)[keep_r]]).astype(np.int64)
    order = np.lexsort((pr, pl))
    pl, pr = pl[order], pr[order]
    dup = np.flatnonzero((pl[1:] == pl[:-1]) & (pr[1:] == pr[:-1]))
    if len(dup):
        pl, pr = np.delete(pl, dup), np.delete(pr, dup)
        print(f"  removed {len(dup):,} duplicate pairs", flush=True)
    print(f"merged: {len(pl):,} pairs over {len(np.unique(pl)):,} entities", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    s1_ids = np.char.add(np.full(n_left, "S1-", dtype="U3"),
                         np.char.mod("%d", np.asarray(L.ids).astype(np.int64)))
    rids = P7.right_ids(R)
    starts = np.flatnonzero(np.concatenate(([True], pl[1:] != pl[:-1])))
    ends = np.concatenate((starts[1:], [len(pl)]))
    groups = np.full(n_left, "", dtype=object)
    for a, b in zip(starts, ends):
        groups[pl[a]] = ",".join(rids[pr[a:b]])
    P7.write_tsv(os.path.join(out_dir, "matching_results.tsv"), s1_ids, groups)
    print(f"wrote {out_dir}/matching_results.tsv", flush=True)

    if not args.no_candidates:
        cl = np.concatenate([np.asarray(bl), np.asarray(rl)]).astype(np.int64)
        cr = np.concatenate([np.asarray(br), np.asarray(rr)]).astype(np.int64)
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

    if args.backup:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            src = os.path.join(P7.OUT, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, f"prev_{name}"))
                print(f"  backed up {name} -> {out_dir}/prev_{name}", flush=True)


if __name__ == "__main__":
    main()
