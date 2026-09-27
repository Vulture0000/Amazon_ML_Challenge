"""
common.py -- shared plumbing: right-side (S2+S3) concatenation, id<->row maps,
ground-truth loading and the competition metric (per-entity macro F0.5).
"""

import os

import numpy as np

from dataio import DATA_DIR, PREP, ROOT, Source, load_country_table  # noqa: F401

WORK = os.path.join(ROOT, "work")


# --------------------------------------------------------------------------
# right side = source 2 ++ source 3 of a split
# --------------------------------------------------------------------------

class Right:
    """Concatenated S2 + S3 view of a split, with the arrays blocking needs."""

    def __init__(self, split, mmap=True):
        self.s2 = Source(f"{split}_s2", mmap=mmap)
        self.s3 = Source(f"{split}_s3", mmap=mmap)
        self.n2 = len(self.s2)
        self.n3 = len(self.s3)
        self.n = self.n2 + self.n3
        cat = lambda a, b: np.concatenate([np.asarray(a), np.asarray(b)])
        self.ids = cat(self.s2.ids, self.s3.ids)
        self.ctry = cat(self.s2.ctry, self.s3.ctry)
        self.flag = cat(self.s2.flag, self.s3.flag)
        self.hnum = cat(self.s2.hnum, self.s3.hnum)
        self.zpc = cat(self.s2.zpc, self.s3.zpc)

        def cat_csr(o1, t1, o2, t2):
            off = np.concatenate([o1, o1[-1] + o2[1:]])
            return off.astype(np.int32), np.concatenate([t1, t2])
        self.ntok_off, self.ntok = cat_csr(self.s2.ntok_off, self.s2.ntok,
                                           self.s3.ntok_off, self.s3.ntok)
        self.atok_off, self.atok = cat_csr(self.s2.atok_off, self.s2.atok,
                                           self.s3.atok_off, self.s3.atok)
        self.npre_off, self.npre = cat_csr(self.s2.npre_off, self.s2.npre,
                                           self.s3.npre_off, self.s3.npre)
        self._ablob_off = None
        self._sorted = None

    def src(self, r):
        return np.where(r < self.n2, 2, 3)

    def row_in_src(self, r):
        return np.where(r < self.n2, r, r - self.n2)

    def name(self, r):
        s = self.s2 if r < self.n2 else self.s3
        return s.name(r if r < self.n2 else r - self.n2)

    def names(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty(len(rows), dtype=object)
        m2 = rows < self.n2
        if m2.any():
            out[m2] = self.s2.names(rows[m2])
        if (~m2).any():
            out[~m2] = self.s3.names(rows[~m2] - self.n2)
        return out.tolist()

    def addrs(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty(len(rows), dtype=object)
        m2 = rows < self.n2
        if m2.any():
            out[m2] = self.s2.addrs(rows[m2])
        if (~m2).any():
            out[~m2] = self.s3.addrs(rows[~m2] - self.n2)
        return out.tolist()

    def lookup_ids(self, numbers):
        """Map entity-id numbers to right rows (sorted search)."""
        if self._sorted is None:
            order = np.argsort(self.ids, kind="stable")
            self._sorted = (self.ids[order], order.astype(np.int64))
        vals, order = self._sorted
        pos = np.searchsorted(vals, numbers)
        pos_c = np.clip(pos, 0, len(vals) - 1)
        ok = vals[pos_c] == numbers
        return order[pos_c], ok


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

def gt_key():
    """Sorted packed keys (s1_row << 24 | right_row) for O(log n) pair labelling."""
    path = os.path.join(WORK, "gt_key.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    tl = np.load(os.path.join(WORK, "gt_s1.npy"))
    tr = np.load(os.path.join(WORK, "gt_right.npy"))
    key = np.sort(tl.astype(np.int64) << 24 | tr)
    np.save(path, key)
    return key


def label_pairs(left_rows, right_rows, key=None):
    """1 where (left_row, right_row) is a ground-truth positive."""
    if key is None:
        key = gt_key()
    left_rows = np.asarray(left_rows, dtype=np.int64)
    right_rows = np.asarray(right_rows, dtype=np.int64)
    sk = left_rows << 24 | right_rows
    pos = np.clip(np.searchsorted(key, sk), 0, len(key) - 1)
    return (np.asarray(key)[pos] == sk).astype(np.int8)


def load_gt(split="train"):
    """Return (s1_row, right_row) int64 arrays of all positive pairs."""
    assert split == "train"
    path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    s1 = Source("train_s1")
    right = Right("train")
    s1_ids = np.asarray(s1.ids)
    order = np.argsort(s1_ids, kind="stable")
    s1_sorted = s1_ids[order]

    # one streaming pass over the ground-truth file
    s1_num, s1_cnt, src_all, num_all = [], [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            s1_num.append(int(a[3:]))
            if b:
                ms = b.split(",")
                s1_cnt.append(len(ms))
                for m in ms:
                    src_all.append(m[0])
                    num_all.append(int(m[3:]))
            else:
                s1_cnt.append(0)
    s1_num = np.array(s1_num, dtype=np.int64)
    s1_cnt = np.array(s1_cnt, dtype=np.int64)
    src_all = np.array(src_all)
    num_all = np.array(num_all, dtype=np.int64)

    pos = np.clip(np.searchsorted(s1_sorted, s1_num), 0, len(s1_sorted) - 1)
    assert np.all(s1_sorted[pos] == s1_num), "S1 ids not found"
    s1_row = order[pos].repeat(s1_cnt)

    is2 = src_all == "2"
    r2, ok2 = right.lookup_ids(num_all[is2])
    r3, ok3 = right.lookup_ids(num_all[~is2])
    assert ok2.all() and ok3.all(), (ok2.mean(), ok3.mean())
    right_row = np.empty(len(num_all), dtype=np.int64)
    right_row[is2] = r2
    right_row[~is2] = r3
    return s1_row.astype(np.int64), right_row


def gt_dict(s1_row, right_row, n_left):
    """Group positive pairs by left row into a dict of numpy arrays."""
    order = np.argsort(s1_row, kind="stable")
    sr = s1_row[order]
    rr = right_row[order]
    bounds = np.searchsorted(sr, np.arange(n_left + 1))
    return {i: rr[bounds[i]:bounds[i + 1]] for i in np.unique(sr)}


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------

def entity_fbeta(tp, npred, ntrue, beta=0.5):
    """Per-entity F_beta with the competition's singleton convention.

    * no true match and no prediction -> 1.0
    * no true match but a prediction   -> 0.0
    """
    p = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    r = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0)
    b2 = beta * beta
    denom = b2 * p + r
    f = np.where(denom > 0, (1 + b2) * p * r / np.maximum(denom, 1e-12), 0.0)
    f = np.where((ntrue == 0) & (npred == 0), 1.0, f)
    f = np.where((ntrue == 0) & (npred > 0), 0.0, f)
    return f


def score(pl, pr, tl, tr, n_left, n_right, beta=0.5):
    """Vectorized macro/per-entity F_beta.

    pl/pr : predicted pairs (left row, right row)
    tl/tr : true pairs
    All left rows in [0, n_left) are scored, including entities with no
    prediction and no true match (singletons).
    """
    pl = np.asarray(pl, np.int64)
    pr = np.asarray(pr, np.int64)
    tl = np.asarray(tl, np.int64)
    tr = np.asarray(tr, np.int64)
    n = int(n_left)
    npred = np.bincount(pl, minlength=n) if len(pl) else np.zeros(n, np.int64)
    ntrue = np.bincount(tl, minlength=n) if len(tl) else np.zeros(n, np.int64)
    a = np.unique(pl * np.int64(n_right) + pr)
    b = np.unique(tl * np.int64(n_right) + tr)
    inter = np.intersect1d(a, b, assume_unique=True)
    tp = (np.bincount(inter // n_right, minlength=n)
          if len(inter) else np.zeros(n, np.int64))
    f = entity_fbeta(tp, npred, ntrue, beta)
    TP = int(tp.sum())
    P = TP / max(int(npred.sum()), 1)
    R = TP / max(int(ntrue.sum()), 1)
    b2 = beta * beta
    micro = (1 + b2) * P * R / (b2 * P + R) if (P + R) > 0 else 0.0
    return {"macro_fbeta": float(f.mean()),
            "micro_fbeta": float(micro),
            "micro_precision": float(P), "micro_recall": float(R),
            "n_entities": n, "tp": TP,
            "pred_pairs": int(npred.sum()), "true_pairs": int(ntrue.sum()),
            "singletons": int((ntrue == 0).sum()),
            "singleton_correct": int(((ntrue == 0) & (npred == 0)).sum()),
            "singleton_fp": int(((ntrue == 0) & (npred > 0)).sum()),
            }


def fmt(name, m):
    return (f"{name:<22s} macroF0.5={m['macro_fbeta']:.4f}  microF0.5={m['micro_fbeta']:.4f}"
            f"  P={m['micro_precision']:.4f} R={m['micro_recall']:.4f}"
            f"  pairs={m['pred_pairs']} singletons_ok={m['singleton_correct']}/{m['singletons']}")
