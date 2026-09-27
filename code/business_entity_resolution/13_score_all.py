"""Score every training candidate with the deployed stage-2 model.

Only the held-out 4.02M pairs were ever scored, because the competition
features that matter most (who else claims this right record?) need scores for
all 40.2M candidates to be computed at all.  One pass, chunked, ~6 min.
"""
import importlib
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M6 = importlib.import_module("06_model")
from common import WORK, Right, Source

OUT = os.path.join(WORK, "pairs_train_sv_all.npy")


def main():
    L, R = Source("train_s1"), Right("train")
    d = M6.load("train", keys=("l", "r", "f", "fz"))
    nzf = d["f"].shape[1]
    n = len(d["l"])
    with open(os.path.join(WORK, "model2_train.pkl"), "rb") as f:
        m = pickle.load(f)["model"]
    out = np.empty(n, np.float32)
    t0 = time.time()
    for b0 in range(0, n, 2_000_000):
        b1 = min(b0 + 2_000_000, n)
        out[b0:b1] = m.predict_proba(
            M6.gather(d, np.arange(b0, b1), nzf))[:, 1]
        print(f"  {b1:,}/{n:,}  {time.time()-t0:.0f}s", flush=True)
    np.save(OUT, out)
    print(f"saved {OUT} in {time.time()-t0:.0f}s", flush=True)
    # quick sanity: the held-out subset must reproduce the stored scores
    sv = np.asarray(d.get("sv")) if "sv" in d else None
    if sv is not None:
        rng = np.random.default_rng(M6.SEED)
        isv = np.zeros(len(L), bool)
        isv[rng.choice(len(L), int(len(L) * M6.VALID_FRAC), replace=False)] = True
        k = isv[np.asarray(d["l"]).astype(np.int64)]
        print(f"max |new - stored| on the held-out rows: "
              f"{np.abs(out[k] - np.asarray(sv)).max():.2e}", flush=True)


if __name__ == "__main__":
    main()
