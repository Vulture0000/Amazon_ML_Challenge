"""Score the test pools: stage-2b on the base pool, stage 2 on the recall pools.

The validation split settled the division of labour: the competition ranker is
+0.0108 macro on the base pool, where its claim/crowd/margin columns are
meaningful, but it has only ever been trained on base-pool rows, so on the recall
pools -- by construction the candidates the first pass rejected -- it collapses
and their thresholds are pushed to 0.99.  The recall pools therefore keep the
stage-2 model they were tuned with, and only the base pool is re-ranked.

Because the stage-2b features describe competition *within the whole candidate
universe*, the base pool's X2 is built over the union of the base pool and both
recall pools, exactly as it was in training.

Run:  python 16_test_score.py
"""
import importlib
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M6 = importlib.import_module("06_model")
S2 = importlib.import_module("stage2b")
from common import WORK, Right, Source

# the deployed test recovery pool is tagged _recv; the train pool that the
# thresholds were fitted on is _recv2 (same configuration, different run)
TAGS = {"test": ["_recv", "_conj"], "train": ["_recv2", "_conj"]}


def main():
    t0 = time.time()
    split = "test"
    tags = TAGS[split]
    L, R = Source(f"{split}_s1"), Right(split)
    with open(os.path.join(WORK, "model2_train.pkl"), "rb") as f:
        v1 = pickle.load(f)
    with open(os.path.join(WORK, "model2b_train.pkl"), "rb") as f:
        v2 = pickle.load(f)
    base_thr, n_x2 = float(v1["thr"]), int(v2["n_x2"])
    d = M6.load(split, keys=("l", "r", "sc", "f", "fz"))
    nzf = d["f"].shape[1]
    l = np.asarray(d["l"]).astype(np.int64)
    r = np.asarray(d["r"]).astype(np.int64)
    sc = np.asarray(d["sc"]).astype(np.float32)
    print(f"base pool {len(l):,} pairs, {len(L):,} left rows", flush=True)

    print("stage 2 (v1) scores for the recall pools", flush=True)
    sv1 = [np.load(os.path.join(WORK, f"pairs_{split}_sv.npy")).astype(np.float32)]
    for tag in tags:
        sv1.append(S2.load_v1(split, tag, d, nzf, v1["model"]))
        print(f"  {tag}: {len(sv1[-1]):,} rows  {time.time()-t0:.0f}s", flush=True)

    print("building the stage-2b features over the union", flush=True)
    pools_l = [l]
    pools_r = [r]
    pools_sc = [sc]
    for tag in tags:
        pools_l.append(S2.pool(split, tag)[0])
        pools_r.append(S2.pool(split, tag)[1])
        pools_sc.append(S2.pool(split, tag)[3])
    lu = np.concatenate(pools_l)
    ru = np.concatenate(pools_r)
    scu = np.concatenate(pools_sc)
    svu = np.concatenate(sv1)
    bounds = np.append(np.cumsum([0] + [len(s) for s in sv1]), len(lu))
    print(f"  universe {len(lu):,} rows", flush=True)
    X2 = S2.extras(lu, ru, scu, svu, R, base_thr)
    del lu, ru, scu, svu, pools_l, pools_r, pools_sc
    xb = X2[bounds[0]:bounds[1]]
    np.save(os.path.join(WORK, f"pairs_{split}_x2.npy"), xb)
    for i, tag in enumerate(tags):
        np.save(os.path.join(WORK, f"pairs_{split}{tag}_x2.npy"),
                X2[bounds[i + 1]:bounds[i + 2]])
    del X2
    print(f"  X2 written  {time.time()-t0:.0f}s", flush=True)

    print("stage-2b (v2) scores for the base pool", flush=True)
    s2 = S2.score(d, np.arange(len(xb)), xb, nzf, v2["model"], n_x2)
    np.save(os.path.join(WORK, f"pairs_{split}_sv2.npy"), s2)
    for t in (0.5, 0.7, 0.81, 0.9):
        k = s2 >= t
        print(f"  >= {t:.2f}: {int(k.sum()):,} pairs, "
              f"{len(np.unique(l[k])):,} entities", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
