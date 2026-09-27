"""Final inference: score test candidates, threshold, write the two required TSVs."""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
F = importlib.import_module("05_fuzzy")
import importlib
M6 = importlib.import_module("06_model")
import common
from common import WORK, Right, Source

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "output"))


def right_ids(R):
    pre = np.where(np.arange(R.n) < R.n2, "S2-", "S3-").astype("U3")
    return np.char.add(pre, np.char.mod("%d", np.asarray(R.ids).astype(np.int64)))


def write_tsv(path, left_ids, groups):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n" if "results" in path
                else "source1_entity_id\tcandidate_entity_ids\n")
        f.write("".join(f"{a}\t{b}\n" for a, b in zip(left_ids, groups)))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    thr_override = os.environ.get("THRESHOLD")
    L = Source(f"{split}_s1")
    R = Right(split)
    n_left = len(L)
    d = M6.load(split)
    l = np.asarray(d["l"]).astype(np.int64)
    r = np.asarray(d["r"]).astype(np.int64)
    with open(os.path.join(WORK, f"model2_train.pkl"), "rb") as fh:
        mdl = pickle.load(fh)
    thr = float(thr_override) if thr_override else float(mdl["thr"])
    nf = d["f"].shape[1]
    print(f"{split}: {len(l):,} pairs scored with model2_train.pkl, thr={thr:.6f}",
          flush=True)
    s = np.empty(len(l), np.float32)
    for b0 in range(0, len(l), 2_000_000):
        b1 = min(b0 + 2_000_000, len(l))
        s[b0:b1] = mdl["model"].predict_proba(
            M6.gather(d, np.arange(b0, b1), nf))[:, 1]
    np.save(os.path.join(WORK, f"pairs_{split}_sv.npy"), s)
    keep = s >= thr
    print(f"  predicted pairs={int(keep.sum()):,} "
          f"({keep.mean()*100:.3f}%)  entities with >=1 match="
          f"{len(np.unique(l[keep])):,}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    s1_ids = np.char.add(np.full(n_left, "S1-", dtype="U3"),
                         np.char.mod("%d", np.asarray(L.ids).astype(np.int64)))
    rids = right_ids(R)
    order = np.lexsort((r, l))
    for name, sel in (("matching_results", keep), ("candidate_pairs", None)):
        t0 = time.time()
        use = np.ones(len(l), bool) if sel is None else sel
        o = order[use[order]]
        lo, ro = l[o], r[o]
        starts = np.flatnonzero(np.concatenate(([True], lo[1:] != lo[:-1])))
        ends = np.concatenate((starts[1:], [len(lo)]))
        out_l = np.full(n_left, "", dtype=object)
        if len(starts):
            for i, (a, b) in enumerate(zip(starts, ends)):
                out_l[lo[a]] = ",".join(rids[ro[a:b]])
        path = os.path.join(OUT, f"{name}.tsv")
        write_tsv(path, s1_ids, out_l)
        print(f"  wrote {path} ({n_left:,} rows, {time.time()-t0:.0f}s)", flush=True)

    if split == "train":
        y = np.asarray(d["y"])
        m = common.score(l[keep], r[keep], l[y == 1], r[y == 1], n_left, R.n)
        print(common.fmt("PREDICTED", m), flush=True)


if __name__ == "__main__":
    main()
