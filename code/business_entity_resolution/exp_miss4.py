"""Size a second recovery pass: which already-predicted entities own the
missed pairs, and how low is their current confidence?

The first recovery pass only touches entities that predict nothing, which is
55.8% of the missed pairs.  The rest belong to entities that already predict,
so adding candidates there can only help if the new candidate outranks the
existing one.  This script buckets those entities by their current best
stage-2 score to see how small a "low confidence" set can be while still
covering most of the remaining missed pairs.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
M6 = importlib.import_module("06_model")
import common
from common import WORK, Source


def main():
    d = M6.load("train", keys=("l", "r", "y", "sv"))
    L = Source("train_s1")
    n_all = len(L)
    l = np.asarray(d["l"]).astype(np.int64)
    y = np.asarray(d["y"]).astype(np.float64)
    sv = np.asarray(d["sv"]).astype(np.float64)
    rng = np.random.default_rng(M6.SEED)
    is_valid = np.zeros(n_all, bool)
    is_valid[rng.choice(n_all, int(n_all * M6.VALID_FRAC), replace=False)] = True
    n_val = int(is_valid.sum())
    remap = np.full(n_all, -1, np.int64)
    remap[np.flatnonzero(is_valid)] = np.arange(n_val)
    va = is_valid[l]
    lv, yv, svv = remap[l[va]], y[va], sv  # sv already holds validation rows only

    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    tl = remap[tl[mm]]
    ntrue = np.bincount(tl, minlength=n_val).astype(np.int64)
    best = np.zeros(n_val)
    np.maximum.at(best, lv, svv)
    has_pred = best > 0

    cov = np.zeros(n_val, np.int64)
    np.add.at(cov, tl, 1)
    got = np.zeros(n_val, np.int64)
    np.add.at(got, lv[yv == 1], 1)
    miss = cov - got
    print(f"validation entities={n_val:,} missed pairs={int(miss.sum()):,}")
    print(f"  owned by entities with a prediction : "
          f"{int(miss[has_pred].sum()):,} over {int((miss[has_pred]>0).sum()):,} entities")
    print(f"  owned by entities without a prediction: "
          f"{int(miss[~has_pred].sum()):,} over {int((miss[~has_pred]>0).sum()):,} entities")

    hp = np.flatnonzero(has_pred)
    order = hp[np.argsort(best[hp])]
    tot = int(miss[has_pred].sum())
    print(f"\nconfidence buckets over the {len(hp):,} predicted entities "
          f"(ascending best stage-2 score):")
    cum = 0
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
        k = int(len(order) * frac)
        sel = order[:k]
        cum = int(miss[sel].sum())
        print(f"  lowest {frac:5.0%} = {k:>7,} entities (best score < "
              f"{best[sel[-1]]:.4f}): missed pairs {cum:>7,} ({cum/tot:.3f} of that group)")

    for cut in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        sel = np.flatnonzero(has_pred & (best < cut))
        print(f"  best score < {cut:<5.2f}: {len(sel):>7,} entities, "
              f"missed pairs {int(miss[sel].sum()):>7,} ({miss[sel].sum()/tot:.3f}), "
              f"true matches {int(ntrue[sel].sum()):,}")

    # a second pass can only *help* entities whose current prediction is wrong;
    # entities that already predict correctly can only lose from extra pairs.
    ncorrect = np.bincount(lv[yv == 1], minlength=n_val)
    wrong = has_pred & (ncorrect == 0)
    print(f"\nvalidation entities with a prediction but no correct pair: "
          f"{int(wrong.sum()):,}  missed pairs {int(miss[wrong].sum()):,} "
          f"({miss[wrong].sum()/tot:.3f} of that group)")
    for cut in (0.5, 0.7, 0.9, 0.99, 1.01):
        sel = np.flatnonzero(wrong & (best < cut))
        print(f"  wrongly predicted and best score < {cut:<5.2f}: {len(sel):>7,} entities, "
              f"missed pairs {int(miss[sel].sum()):,}, true matches {int(ntrue[sel].sum()):,}")
    part = np.flatnonzero(wrong & (miss > 0))
    print(f"  wrongly predicted and owning a missed pair: {len(part):,} entities, "
          f"{int(miss[part].sum()):,} missed pairs -> a second pass could recover at "
          f"most {miss[part].sum()/max(miss.sum(),1):.3f} of all missed pairs")


if __name__ == "__main__":
    main()
