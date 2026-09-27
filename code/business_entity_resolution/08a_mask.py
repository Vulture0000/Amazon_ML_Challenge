"""Build the left-entity mask for the recovery pass.

The recovery pass is only allowed to touch entities the first pass predicts
nothing for, so the mask is "no candidate scored at or above the stage-2
threshold".  On train that is derived from the saved validation-split stage-2
scores (work/pairs_train_sv.npy) so the effect can be measured honestly; on
test it is derived from the predictions that 07_predict.py already wrote.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
M6 = importlib.import_module("06_model")
from common import WORK, Source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.0, help="0 -> use the model threshold")
    ap.add_argument("--valid-only", action="store_true",
                    help="train: restrict to the validation entities")
    ap.add_argument("--all-valid", action="store_true",
                    help="train: every validation entity, predicted or not")
    ap.add_argument("--all-rows", action="store_true",
                    help="every row of the split: the target set for a global pass, "
                         "which must not depend on a previous output file")
    args = ap.parse_args()
    split = args.split
    L = Source(f"{split}_s1")
    n = len(L)
    mask = np.zeros(n, bool)

    if args.all_rows:
        mask[:] = True
        print(f"{split}: all {n:,} rows selected")
    elif split == "train":
        d = M6.load(split, keys=("l", "sv"))
        l = np.asarray(d["l"]).astype(np.int64)
        sv = np.asarray(d["sv"])
        rng = np.random.default_rng(M6.SEED)
        is_valid = np.zeros(n, bool)
        is_valid[rng.choice(n, int(n * M6.VALID_FRAC), replace=False)] = True
        thr = args.thr
        if thr <= 0:
            import pickle
            with open(os.path.join(WORK, "model2_train.pkl"), "rb") as f:
                thr = pickle.load(f)["thr"]
        print(f"threshold={thr:.6f}  validation entities={int(is_valid.sum()):,}")
        have = np.zeros(n, bool)
        va = is_valid[l]
        have[l[va][sv >= thr]] = True
        if args.all_valid:
            mask = is_valid.copy()
        elif args.valid_only:
            mask = is_valid & ~have
        else:
            mask = ~have
        print(f"entities with no prediction: {int((~have).sum()):,}")
    else:
        import csv
        pred = os.path.join(os.path.dirname(WORK), "output", "matching_results.tsv")
        order = np.argsort(np.asarray(L.ids), kind="stable")
        vals = np.asarray(L.ids)[order]
        seen = []
        with open(pred, newline="") as f:
            rd = csv.reader(f, delimiter="\t")
            next(rd)
            for row in rd:
                if row[1].strip():  # the TSV lists every S1 entity, empty when unmatched
                    seen.append(int(row[0][3:]))
        nums = np.array(seen, np.int64)
        p = np.clip(np.searchsorted(vals, nums), 0, len(vals) - 1)
        have = np.zeros(n, bool)
        have[order[p[vals[p] == nums]]] = True
        mask = ~have
        print(f"rows with at least one match: {len(nums):,}  "
              f"entities with at least one prediction: {int(have.sum()):,} "
              f"({len(nums):,} matched ids)")

    np.save(args.out, mask)
    print(f"{args.out}: {int(mask.sum()):,} of {n:,} entities ({mask.mean():.2%})")


if __name__ == "__main__":
    main()
