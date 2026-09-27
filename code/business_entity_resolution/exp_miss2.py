"""Which extra key families would recover the pairs the current blocking misses?

Drives the same per-family agreement measurement as exp_keys.py, but over the
*actual* missed pairs of the full production run, so the marginal value of each
candidate family is measured against what the six live families already cover.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erlib as E
from common import DATA_DIR, WORK, Right, Source

DIGIT2LET = str.maketrans({"0": "o", "1": "l", "2": "z", "3": "e", "4": "a",
                           "5": "s", "6": "b", "7": "t", "8": "b", "9": "g"})
VOW = re.compile(r"[aeiou]")
N_MISS = int(os.environ.get("N_MISS", "120000"))


def skel(tok):
    return VOW.sub("", tok.translate(DIGIT2LET))[:6]


def main():
    L = Source("train_s1")
    R = Right("train")
    tl, tr = np.load(os.path.join(WORK, "gt_s1.npy")), np.load(os.path.join(WORK, "gt_right.npy"))
    l = np.load(os.path.join(WORK, "pairs_train_l.npy")).astype(np.int64)
    r = np.load(os.path.join(WORK, "pairs_train_r.npy")).astype(np.int64)
    y = np.load(os.path.join(WORK, "pairs_train_y.npy"))
    W = np.int64(1) << 24
    got = np.unique(l[y == 1].astype(np.int64) * W + r[y == 1])
    allp = np.unique(tl.astype(np.int64) * W + tr)
    miss = np.setdiff1d(allp, got, assume_unique=True)
    print(f"true pairs={len(allp):,}  kept={len(got):,}  missed={len(miss):,} "
          f"({len(miss)/len(allp):.3f})", flush=True)
    rng = np.random.default_rng(0)
    take = rng.choice(len(miss), min(N_MISS, len(miss)), replace=False)
    ml = (miss[take] // W).astype(np.int64)
    mr = (miss[take] % W).astype(np.int64)

    s1 = L.ids
    rids = R.ids
    src = R.src(mr)
    lkey = np.char.add("S1-", s1[ml].astype("U10"))
    rkey = np.char.add(np.char.add("S", src.astype("U1") + "-"), rids[mr].astype("U10"))
    want = {}
    for i in range(len(ml)):
        want[lkey[i]] = None
        want[rkey[i]] = None
    df1 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"),
                      sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    df2 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"),
                      sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    df3 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"),
                      sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    allrec = pd.concat([df1, df2, df3])
    sub = allrec.loc[list(want)]
    print(f"rows to featurize: {len(sub):,}", flush=True)

    def feats(row):
        nn, nc, _ = E.norm_name(row.business_name)
        an, ac, _ = E.norm_addr(row.business_address, row.country)
        nt = nc.split()
        at = [t for t in ac.split() if t != "unit"]
        return (nn, set(nt), set(at), set(t for t in at if t.isdigit()),
                set(map(skel, nt)), set(map(skel, at)))

    cache = {}
    for i, row in sub.iterrows():
        cache[i] = feats(row)
    print(f"featurized {len(cache):,}", flush=True)

    keys = ["nset", "aset", "npre", "nsuf", "nskel", "nskelset", "askel",
            "atok", "hnum", "n_atok", "n_askel", "hnum_npre",
            "npre_nsuf", "hnum_askel", "nn_full"]
    hit = dict.fromkeys(keys, 0)
    shown = 0
    for i in range(len(ml)):
        a = cache[lkey[i]]
        b = cache[rkey[i]]
        f = {
            "nset": a[1] == b[1], "aset": a[2] == b[2],
            "npre": bool({x[:4] for x in a[1]} & {x[:4] for x in b[1]}),
            "nsuf": bool({x[-4:] for x in a[1]} & {x[-4:] for x in b[1]}),
            "nskel": bool(a[4] & b[4]), "nskelset": a[4] == b[4],
            "askel": bool(a[5] & b[5]), "atok": bool(a[2] & b[2]),
            "hnum": bool(a[3] & b[3]),
            "n_atok": bool(a[1] & b[2]), "n_askel": bool(a[4] & b[5]),
            "hnum_npre": bool(a[3] & b[3]) and bool({x[:4] for x in a[1]} & {x[:4] for x in b[1]}),
            "npre_nsuf": bool({x[:4] for x in a[1]} & {x[:4] for x in b[1]})
                          and bool({x[-4:] for x in a[1]} & {x[-4:] for x in b[1]}),
            "hnum_askel": bool(a[3] & b[3]) and bool(a[5] & b[5]),
            "nn_full": a[0] == b[0],
        }
        for k in keys:
            hit[k] += f[k]
        if shown < 12 and (f["nskel"] or f["askel"] or f["nsuf"]):
            print(f"  MISS {lkey[i]} | {rkey[i]}"
                  f"\n    {a[0]!r}  ||  {b[0]!r}\n    {sorted(a[2])[:6]} || {sorted(b[2])[:6]}"
                  f"  skel_n={sorted(a[4] & b[4])} skel_a={sorted(a[5] & b[5])}")
            shown += 1
    n = len(ml)
    print(f"\nmarginal recovery over {n:,} missed pairs:")
    for k in sorted(keys, key=lambda k: -hit[k]):
        print(f"  {k:<10s} {hit[k]/n:7.4f}  ({hit[k]:,})")
    cur = set(["nset", "npre", "atok", "hnum"])
    for combo in (["nsuf"], ["askel"], ["nsuf", "askel"], ["nsuf", "askel", "nskel"],
                  ["nsuf", "askel", "nskel", "n_atok"], ["nn_full", "nsuf", "askel"]):
        c = 0
        for i in range(n):
            a = cache[lkey[i]]
            b = cache[rkey[i]]
            if "nn_full" in combo and a[0] == b[0]:
                c += 1
                continue
            gotit = False
            if "nsuf" in combo and {x[-4:] for x in a[1]} & {x[-4:] for x in b[1]}:
                gotit = True
            if not gotit and "askel" in combo and (a[5] & b[5]):
                gotit = True
            if not gotit and "nskel" in combo and (a[4] & b[4]):
                gotit = True
            if not gotit and "n_atok" in combo and (a[1] & b[2]):
                gotit = True
            c += gotit
        print(f"  UNION {str(combo):<44s} {c/n:7.4f}  ({c:,})")


if __name__ == "__main__":
    main()
