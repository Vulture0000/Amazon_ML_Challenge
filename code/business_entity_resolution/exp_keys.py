"""Measure per-family key agreement on TRUE pairs (blocking recall ceiling)."""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erlib as E
from common import DATA_DIR

DIGIT2LET = str.maketrans({"0": "o", "1": "l", "2": "z", "3": "e", "4": "a",
                           "5": "s", "6": "b", "7": "t", "8": "b", "9": "g"})
VOW = re.compile(r"[aeiou]")


def skel(tok):
    t = tok.translate(DIGIT2LET)
    t = VOW.sub("", t)
    return t[:6]


def load(path, ids=None):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df.set_index("entity_id")
    if ids is not None:
        df = df.loc[[i for i in ids if i in df.index]]
    return df


rng = np.random.default_rng(0)
gt = pd.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"),
                 sep="\t", dtype=str, keep_default_na=False)
sel = gt.sample(30000, random_state=0)
sel = sel[sel.matched_entity_ids.str.strip() != ""]
wanted = set(sel.source1_entity_id)
for v in sel.matched_entity_ids:
    wanted.update(v.split(","))
print("entities:", len(sel), "rows to read:", len(wanted))

s1 = load(os.path.join(DATA_DIR, "train", "train_source1.tsv"))
s2 = load(os.path.join(DATA_DIR, "train", "train_source2.tsv"))
s3 = load(os.path.join(DATA_DIR, "train", "train_source3.tsv"))
allrec = pd.concat([s1, s2, s3])
allrec = allrec.loc[[i for i in wanted if i in allrec.index]]
print("loaded", len(allrec))


def feats(row):
    nn, nc, _ = E.norm_name(row.business_name)
    an, ac, _ = E.norm_addr(row.business_address, row.country)
    nt = nc.split()
    at = [t for t in ac.split() if t != "unit"]
    nums = [t for t in at if t.isdigit()]
    return nn, set(nt), set(at), set(nums), set(map(skel, nt)), set(map(skel, at))


cache = {i: feats(r) for i, r in allrec.iterrows()}
print("featurized", len(cache))

import collections
flags = collections.defaultdict(list)
per_pair = []
for _, row in sel.iterrows():
    a = cache[row.source1_entity_id]
    for m in row.matched_entity_ids.split(","):
        b = cache.get(m)
        if b is None:
            continue
        nn2, ntok2, atok2, nums2, sk2, ask2 = b
        f = {}
        f["ntok"] = bool(a[1] & ntok2)
        f["npre"] = bool({x[:4] for x in a[1]} & {x[:4] for x in ntok2})
        f["nsuf"] = bool({x[-4:] for x in a[1]} & {x[-4:] for x in ntok2})
        f["nskel"] = bool(a[4] & sk2)
        f["nset"] = a[1] == ntok2
        f["nskelset"] = a[4] == sk2
        f["atok"] = bool(a[2] & atok2)
        f["askel"] = bool(a[5] & ask2)
        f["aset"] = a[2] == atok2
        ah = [x for x in a[2] if x.isdigit()]
        bh = [x for x in atok2 if x.isdigit()]
        f["hnum"] = bool(ah and bh and ah[0] == bh[0])
        f["hnum_askel"] = f["hnum"] and bool(a[5] & ask2)
        f["hnum_atok"] = f["hnum"] and bool(a[2] & atok2)
        f["zip_atok"] = bool(a[3] & nums2) and bool(a[2] & atok2)
        per_pair.append(f)
print("true pairs:", len(per_pair))
allk = sorted(per_pair[0])
for k in allk:
    v = sum(p[k] for p in per_pair) / len(per_pair)
    print(f"  single {k:<12s} {v:.4f}")
combos = [
    ["nset", "aset", "npre", "askel", "hnum"],
    ["nset", "aset", "npre", "nsuf", "askel", "hnum"],
    ["npre", "askel", "hnum", "nset"],
    ["npre", "nsuf", "nskel", "askel", "atok", "hnum"],
    ["nset", "nskelset", "aset", "askel", "npre", "nsuf", "hnum"],
    ["nset", "aset", "npre", "nsuf", "askel", "hnum", "zip_atok"],
    ["hnum_atok", "npre", "nset", "nskel", "nsuf"],
    ["hnum_atok", "npre", "nset", "nskel", "nsuf", "askel"],
    ["npre", "nset", "aset", "hnum"],
    ["npre", "nset", "aset"],
]
print()
for c in combos:
    u = sum(any(p[k] for k in c) for p in per_pair) / len(per_pair)
    print(f"  union {u:.4f}  {c}")
