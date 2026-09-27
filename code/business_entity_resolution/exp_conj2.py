"""Cap / recall trade-off for the conjunctive (nsuf x askel) family.

The first measurement of this index was wrong: it built the posting array with
02_blocking.build_index(), which packs (key << 24 | row) into an int64, so the
50-bit cross keys overflowed and almost nothing was found.  With a correct
64-bit index the raw volume at cap 20,000 is ~20k candidates per left row,
which is far too much to run globally.  What matters is the trade-off: lower
caps should keep the specific keys that carry the true matches while dropping
the generic ones that generate the noise.  This measures both.
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
B = importlib.import_module("02_blocking")
C9 = importlib.import_module("09_conj")
R8 = importlib.import_module("08_recover")
import common
from common import Right, Source, label_pairs

CAPS = [int(x) for x in os.environ.get("CAPS", "20,50,100,200,500,1000,2000").split(",")]


def main():
    n_left = int(os.environ.get("N_LEFT", "30000"))
    L, R = Source("train_s1"), Right("train")
    noff, nkey = R8.right_csr("train", "nsuf")
    aoff, akey = R8.right_csr("train", "askel")
    r_off, r_key, _ = C9.cross_keys(noff, nkey, aoff, akey)
    inv_keys, inv_rows, kv, kc = C9.build_index64_full(r_key, r_off)
    del r_key, noff, nkey, aoff, akey
    print(f"index: {len(inv_keys):,} postings, {len(kv):,} keys, "
          f"max df={int(kc.max()):,}, mean df={len(inv_keys)/len(kv):.1f}", flush=True)

    rng = np.random.default_rng(1)
    rows = np.sort(rng.choice(len(L), n_left, replace=False))
    keep = np.zeros(len(L), bool)
    keep[rows] = True
    ln = C9.left_csr("train", "nsuf")
    la = C9.left_csr("train", "askel")
    l_off, l_key, _ = C9.cross_keys(*B.filter_csr(ln[0], ln[1], keep),
                                    *B.filter_csr(la[0], la[1], keep))
    tl, tr = common.load_gt("train")
    tkey = (tl.astype(np.int64) << 32) | tr.astype(np.int64)
    tset = set((tl.astype(np.int64) * 20000000 + tr.astype(np.int64)).tolist())
    del tkey
    print(f"{n_left:,} left rows, {len(l_key):,} left cross keys "
          f"({len(l_key)/n_left:.1f}/row)", flush=True)
    for cap in CAPS:
        res = B.expand(l_off, l_key, inv_keys, inv_rows, kv, kc, cap, 1,
                       max_out=2_000_000_000)
        if res is None:
            print(f"  cap={cap:<6d} raw=0", flush=True)
            continue
        cl, cr = res
        raw = len(cl)
        hit = np.fromiter(((int(a) * 20000000 + int(b)) in tset for a, b in zip(cl, cr)),
                          bool, count=min(raw, 4_000_000)) if raw <= 4_000_000 else None
        rec = f"{hit.sum()/max(n_left,1):.3f} true/row" if hit is not None else "n/a"
        print(f"  cap={cap:<6d} raw={raw:>13,} = {raw/n_left:9.1f}/row  "
              f"-> {raw/n_left*1732544/1e9:6.2f}B test   {rec}", flush=True)


if __name__ == "__main__":
    main()
