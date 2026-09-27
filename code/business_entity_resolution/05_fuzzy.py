"""Stage 2 features: string-similarity metrics (rapidfuzz) for the kept pairs."""
import os
import re
import sys
import time

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, Right, Source

CHUNK = int(os.environ.get("FUZZ_CHUNK", "2000000"))
TAG = os.environ.get("TAG", "")
DIGITS = re.compile(r"\d+")

FUZZ_NAMES = ["n_jw", "n_wratio", "n_tset", "n_tsort",
              "a_jw", "a_wratio", "a_tset", "a_tsort",
              "n_lenr", "a_lenr", "n_digit_eq", "a_first_eq"]


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    L = Source(f"{split}_s1")
    R = Right(split)
    l = np.load(os.path.join(WORK, f"pairs_{split}{TAG}_l.npy"))
    r = np.load(os.path.join(WORK, f"pairs_{split}{TAG}_r.npy"))
    n = len(l)
    print(f"{split}{TAG}: {n:,} kept pairs, {len(FUZZ_NAMES)} fuzzy features", flush=True)
    workers = int(os.environ.get("WORKERS", "10"))
    out = np.empty((n, len(FUZZ_NAMES)), np.float32)
    t0 = time.time()
    for c0 in range(0, n, CHUNK):
        c1 = min(c0 + CHUNK, n)
        ln, rn = l[c0:c1].astype(np.int64), r[c0:c1].astype(np.int64)
        ul, il = np.unique(ln, return_inverse=True)
        ur, ir = np.unique(rn, return_inverse=True)
        ln_s = L.names(ul)
        la_s = L.addrs(ul)
        rn_s = R.names(ur)
        ra_s = R.addrs(ur)
        an = [ln_s[i] if ln_s[i] else " " for i in il]
        bn = [rn_s[i] if rn_s[i] else " " for i in ir]
        aa = [la_s[i] if la_s[i] else " " for i in il]
        ba = [ra_s[i] if ra_s[i] else " " for i in ir]
        o = c0
        out[o:o + len(an), 0] = _cp(an, bn, JaroWinkler.similarity, workers) / 100.0
        out[o:o + len(an), 1] = _cp(an, bn, fuzz.WRatio, workers) / 100.0
        out[o:o + len(an), 2] = _cp(an, bn, fuzz.token_set_ratio, workers) / 100.0
        out[o:o + len(an), 3] = _cp(an, bn, fuzz.token_sort_ratio, workers) / 100.0
        out[o:o + len(aa), 4] = _cp(aa, ba, JaroWinkler.similarity, workers) / 100.0
        out[o:o + len(aa), 5] = _cp(aa, ba, fuzz.WRatio, workers) / 100.0
        out[o:o + len(aa), 6] = _cp(aa, ba, fuzz.token_set_ratio, workers) / 100.0
        out[o:o + len(aa), 7] = _cp(aa, ba, fuzz.token_sort_ratio, workers) / 100.0
        na = np.fromiter((len(x) for x in an), np.int32, len(an)).clip(1)
        nb = np.fromiter((len(x) for x in bn), np.int32, len(an)).clip(1)
        aa_ = np.fromiter((len(x) for x in aa), np.int32, len(aa)).clip(1)
        ab = np.fromiter((len(x) for x in ba), np.int32, len(aa)).clip(1)
        out[o:o + len(an), 8] = np.minimum(na, nb) / np.maximum(na, nb)
        out[o:o + len(aa), 9] = np.minimum(aa_, ab) / np.maximum(aa_, ab)
        da = ["|".join(DIGITS.findall(x)) for x in an]
        db = ["|".join(DIGITS.findall(x)) for x in bn]
        out[o:o + len(an), 10] = np.fromiter(
            (x == y and x != "" for x, y in zip(da, db)), bool, len(an))
        fa = np.fromiter((x[0] for x in aa), "U1", len(aa))
        fb = np.fromiter((x[0] for x in ba), "U1", len(aa))
        out[o:o + len(aa), 11] = (fa == fb) & (fa != " ")
        el = time.time() - t0
        print(f"  [{c1/n:6.1%}] {c1:,}/{n:,} {el:.0f}s eta={el/max(c1,1)*(n-c1):.0f}s",
              flush=True)
    np.save(os.path.join(WORK, f"pairs_{split}{TAG}_fz.npy"), out)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
