"""Size a *conjunctive* global recovery pass.

61,026 of the missed pairs belong to entities that do predict, so a global pass
has to compete with an existing prediction -- and a disjunctive nsuf|askel pass
over all 1.7M test entities is ~42 billion raw candidates.  A conjunction
(share a name key AND an address key) is far more selective, so this measures
(a) how much of the missed-pair population each single key, the union and the
conjunction cover, and (b) how many candidates per left row the
(nsuf x askel) cross-product index actually produces.
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
B = importlib.import_module("02_blocking")
X = importlib.import_module("02b_extra_keys")
import common
import erlib as E
from common import PREP, WORK, Right, Source

VOW, DIGIT2LET = X.VOW, X.DIGIT2LET
_rid = {}


def read_rows(tag, kind, rows):
    """Decoded text of the given rows, from the preprocessed name/addr blobs."""
    off = np.load(os.path.join(PREP, f"{tag}_{kind}.off.npy"), mmap_mode="r")
    rows = np.asarray(rows, np.int64)
    out = np.empty(len(rows), object)
    with open(os.path.join(PREP, f"{tag}_{kind}.blob"), "rb") as f:
        for i, r in enumerate(rows):
            a, b = int(off[r]), int(off[r + 1])
            f.seek(a)
            out[i] = f.read(b - a).decode("utf8", "replace")
    return out


def keyset(name_txt, addr_txt):
    suf, skel = set(), set()
    for t in E.norm_name(name_txt)[1].split():
        suf.add(int(X.key_id(b"s4", t[-4:])))
    for t in E.norm_addr(addr_txt, "")[1].split():
        if t == "unit":
            continue
        s = VOW.sub("", t.translate(DIGIT2LET))[:6]
        if s:
            skel.add(int(X.key_id(b"ak", s)))
    return suf, skel


def right_csr(split, name):
    o2 = np.load(os.path.join(PREP, f"{split}_s2_{name}.off.npy"))
    o3 = np.load(os.path.join(PREP, f"{split}_s3_{name}.off.npy"))
    k2 = np.load(os.path.join(PREP, f"{split}_s2_{name}.npy"))
    k3 = np.load(os.path.join(PREP, f"{split}_s3_{name}.npy"))
    return (np.concatenate([o2, o2[-1] + o3[1:]]).astype(np.int64),
            np.concatenate([k2, k3]))


def left_csr(split, name, rows):
    off = np.load(os.path.join(PREP, f"{split}_s1_{name}.off.npy"))
    keys = np.load(os.path.join(PREP, f"{split}_s1_{name}.npy"))
    keep = np.zeros(off.shape[0] - 1, bool)
    keep[rows] = True
    return B.filter_csr(off, keys, keep)


def cross_csr(noff, nkey, aoff, akey):
    """Per-row cross product of two key sets, packed into a single int64 key."""
    nn = np.diff(noff).astype(np.int64)
    na = np.diff(aoff).astype(np.int64)
    off = np.zeros(len(nn) + 1, np.int64)
    np.cumsum(nn * na, out=off[1:])
    flat = np.empty(int(off[-1]), np.int64)
    npos, apos, w = 0, 0, 0
    nkv = nkey.astype(np.int64)
    akv = akey.astype(np.int64)
    for i in range(len(nn)):
        for j in range(npos, npos + nn[i]):
            base = nkv[j] << 25
            for k in range(apos, apos + na[i]):
                flat[w] = base | akv[k]
                w += 1
        npos += nn[i]
        apos += na[i]
    return off, flat


def main():
    n_sample = int(os.environ.get("N_MISS", "20000"))
    n_left = int(os.environ.get("N_LEFT", "20000"))
    L, R = Source("train_s1"), Right("train")

    tl, tr = common.load_gt("train")
    cand_l = np.asarray(np.load(os.path.join(WORK, "pairs_train_l.npy"), mmap_mode="r"))
    cand_r = np.asarray(np.load(os.path.join(WORK, "pairs_train_r.npy"), mmap_mode="r"))
    key = (tl.astype(np.int64) << 32) | tr.astype(np.int64)
    ckey = np.sort((cand_l.astype(np.int64) << 32) | cand_r.astype(np.int64))
    pos = np.searchsorted(ckey, key)
    ok = (pos < len(ckey)) & (ckey[np.minimum(pos, len(ckey) - 1)] == key)
    print(f"true pairs={len(tl):,}  in the candidate set={int(ok.sum()):,} "
          f"({ok.mean():.4f})  missed={int((~ok).sum()):,}")
    del ckey, cand_l, cand_r

    rng = np.random.default_rng(0)
    sel = rng.choice(np.flatnonzero(~ok), min(n_sample, int((~ok).sum())), replace=False)
    in2 = tr[sel] < R.n2
    lrow, rrow = tl[sel], tr[sel]
    lname = read_rows("train_s1", "name", lrow)
    laddr = read_rows("train_s1", "addr", lrow)
    rname = np.empty(len(sel), object)
    raddr = np.empty(len(sel), object)
    for store, is2 in (("s2", in2), ("s3", ~in2)):
        u = np.flatnonzero(is2)
        if not len(u):
            continue
        rr = np.where(in2, rrow, rrow - R.n2)[u]
        rname[u] = read_rows(f"train_{store}", "name", rr)
        raddr[u] = read_rows(f"train_{store}", "addr", rr)
    ns = np.zeros(len(sel), bool)
    ak = np.zeros(len(sel), bool)
    for i in range(len(sel)):
        sa, ka = keyset(lname[i], laddr[i])
        sb, kb = keyset(rname[i], raddr[i])
        ns[i] = bool(sa & sb)
        ak[i] = bool(ka & kb)
    n = len(sel)
    print(f"\ncoverage of {n:,} missed pairs:")
    print(f"  nsuf         {ns.mean():.4f}")
    print(f"  askel        {ak.mean():.4f}")
    print(f"  nsuf|askel   {(ns | ak).mean():.4f}")
    print(f"  nsuf&askel   {(ns & ak).mean():.4f}   <- conjunctive family")

    rows = np.sort(rng.choice(len(L), n_left, replace=False))
    rn_off, rn_key = right_csr("train", "nsuf")
    ra_off, ra_key = right_csr("train", "askel")
    rx_off, rx_key = cross_csr(rn_off, rn_key, ra_off, ra_key)
    lx_off, lx_key = cross_csr(*left_csr("train", "nsuf", rows),
                               *left_csr("train", "askel", rows))
    print(f"\ncross keys: left {len(lx_key):,} ({len(lx_key)/n_left:.1f}/row)  "
          f"right {len(rx_key):,} ({len(rx_key)/R.n:.1f}/row)")
    import hashlib
    h = np.array([int.from_bytes(hashlib.blake2b(int(v).to_bytes(8, "little"),
           digest_size=4, key=b"cx").digest(), "little") & 0x1FFFFFF
           for v in np.unique(rx_key)], np.int64)
    uniq, inv = np.unique(rx_key, return_inverse=True)
    rx_h = h[inv]
    del inv, uniq
    idx = B.build_index(rx_h.astype(np.int32), rx_off, R.n)
    kv, kc = B.key_counts(rx_h.astype(np.int32))
    lx_h = np.array([int.from_bytes(hashlib.blake2b(int(v).to_bytes(8, "little"),
           digest_size=4, key=b"cx").digest(), "little") & 0x1FFFFFF for v in lx_key],
           np.int32)
    lx_off2, lx_h, _ = None, lx_h, None
    lx_off, lx_key = lx_off, lx_h
    for cap in (50, 200, 2000, 20000):
        res = B.expand(lx_off, lx_key, *idx, kv, kc, cap, 1, max_out=400_000_000)
        raw = 0 if res is None else len(res[0])
        print(f"  cap={cap:<6d} raw={raw:>12,} = {raw/n_left:8.1f}/left -> "
              f"{raw/n_left*1732544/1e9:6.2f}B for the whole test set")


if __name__ == "__main__":
    main()
