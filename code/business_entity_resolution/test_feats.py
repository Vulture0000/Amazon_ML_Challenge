"""Sanity-check feature quality + cascade ranking on the sample candidate set."""
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FT
from common import WORK, Right, Source

tag = sys.argv[1] if len(sys.argv) > 1 else "trainsm"
L = Source("train_s1")
R = Right("train")
lidx = np.load(os.path.join(WORK, f"cand_{tag}_leftidx.npy"))
Lp = np.load(os.path.join(WORK, f"cand_{tag}_l.npy"))
Rp = np.load(os.path.join(WORK, f"cand_{tag}_r.npy"))
lt = Lp.astype(np.int64)
rt = Rp.astype(np.int64)
ln = lidx[lt]
tl = np.load(os.path.join(WORK, "gt_s1.npy"))
tr = np.load(os.path.join(WORK, "gt_right.npy"))
glt = lidx[lt]
key = np.sort(tl.astype(np.int64) << 24 | tr)
sk = glt.astype(np.int64) << 24 | rt
pos = np.clip(np.searchsorted(key, sk), 0, len(key) - 1)
y = (key[pos] == sk).astype(np.int8)
print(f"pairs={len(y):,}  positives={y.sum():,}  pos_rate={y.mean():.5f}")
print(f"recall of this candidate set: {y.sum():,}")

t0 = time.time()
out, mats = FT.build_features(L, R, ln, rt)
X = FT.stack(out)
print(f"features in {time.time()-t0:.1f}s -> {X.shape}")
for i, k in enumerate(FT.FEATURE_ORDER):
    a = roc_auc_score(y, X[:, i])
    ap = average_precision_score(y, X[:, i])
    print(f"  {k:<12s} auc={a:.4f} ap={ap:.5f}  pos_mean={X[y==1,i].mean():.3f} "
          f"neg_mean={X[y==0,i].mean():.3f}")

for K in (5, 10, 15, 20, 30):
    ordr = np.lexsort((-X[:, 0], lt))
    sl = lt[ordr]
    first = np.ones(len(sl), bool)
    first[1:] = sl[1:] != sl[:-1]
    grp = np.cumsum(first) - 1
    posn = np.arange(len(sl)) - np.flatnonzero(first)[grp]
    keep = ordr[posn < K]
    print(f"topK={K:<3d} kept={len(keep):>9,d}  recall_kept={y[keep].sum():>8,d} "
          f"({y[keep].sum()/max(y.sum(),1):.4f} of cand positives)")
