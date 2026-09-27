"""
Stage 2 -- candidate generation (blocking) with an inverted index.

Every Source-1 record is probed against an inverted index built over the
Source-2 + Source-3 records.  Seven complementary key families are used; each
one is capped by its document frequency so that a generic token ("services",
"main") cannot blow the candidate set up:

  T1 name_token     a single canonical name token
  T2 name_prefix4   4-char prefix of a name token  -> survives 1-char typos
  T3 name_set       the full order-insensitive name token set -> word reorder
  T4 addr_token     a single canonical address token
  T5 addr_set       the full order-insensitive address token set
  T6 house_number   first house number of the address
  T7 zip            postal code (US ZIP / Indian PIN)

Pairs produced by several families are merged, and the family mask is kept so
the classifier can see *how* a candidate was proposed.

    python 02_blocking.py --split train [--sample-left 100000] [--recall]
"""

import argparse
import os
import time

import numpy as np

import common
from common import WORK, Right, Source

MASK = (1 << 24) - 1          # right-row bit budget in the packed pair key
HASH_MASK = (1 << 25) - 1
PRIME = 0x9E3779B97F4A7C15

DEFAULT_CAPS = {
    "name_token": 40,
    "name_prefix4": 25,
    "name_set": 40,
    "addr_token": 12,
    "addr_set": 12,
    "house_number": 60,
    "zip": 60,
    "nsuf": 2000,
    "askel": 2000,
}


def set_key(tokens, off, salt):
    """Order-insensitive set key per row from a CSR of int32 token ids."""
    t = tokens.astype(np.uint64)
    mixed = (t * np.uint64(PRIME)) & np.uint64(0xFFFFFFFF)
    n = len(mixed)
    starts = np.minimum(off[:-1].astype(np.int64), max(n - 1, 0))
    if n:
        sums = np.add.reduceat(mixed, starts)
    else:
        sums = np.zeros(len(off) - 1, np.uint64)
    empty = off[1:] == off[:-1]
    sums[empty] = 0
    return (((sums ^ np.uint64(salt)) * np.uint64(0x2545F4914F6CDD1D)) >> np.uint64(7)
            & np.uint64(HASH_MASK)).astype(np.int64)


def num_key(values, salt, valid):
    v = values.astype(np.uint64)
    out = ((v * np.uint64(0x27220A95)) ^ np.uint64(salt)) & np.uint64(HASH_MASK)
    out[~valid] = 0
    return out.astype(np.int64)


def build_index(keys, off, n):
    """Inverted index: sorted (key, row) pairs packed into one int64 array."""
    cnt = np.diff(off).astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), cnt)
    packed = (keys.astype(np.int64) << 24) | rows
    packed.sort()
    return packed >> 24, (packed & MASK).astype(np.int64)


def key_counts(keys):
    v, c = np.unique(keys, return_counts=True)
    return v, c


def expand(left_off, left_keys, inv_keys, inv_rows, kv, kc, cap, bit,
           left_rows=None, max_out=8_000_000, stats=None):
    """All (left, right) pairs sharing a left key with 1 <= df <= cap."""
    pos = np.searchsorted(kv, left_keys)
    np.clip(pos, 0, max(len(kv) - 1, 0), out=pos)
    ok = (len(kv) > 0) & (kv[pos] == left_keys)
    dfv = np.zeros(len(left_keys), np.int64)
    dfv[ok] = kc[pos[ok]]
    ok &= (dfv >= 1) & (dfv <= cap)
    if not ok.any():
        return
    if left_rows is None:
        left_rows = np.repeat(np.arange(len(left_off) - 1, dtype=np.int64),
                              np.diff(left_off))
    lrow = left_rows[ok]
    kk = left_keys[ok]
    lo = np.searchsorted(inv_keys, kk, side="left")
    hi = np.searchsorted(inv_keys, kk, side="right")
    cnt = hi - lo
    keep = cnt > 0
    lrow, lo, cnt = lrow[keep], lo[keep], cnt[keep]
    if len(cnt) == 0:
        return

    start = 0
    acc_l, acc_r, acc_t = [], [], 0
    while start < len(cnt):
        end = start
        tot = 0
        while end < len(cnt) and tot < max_out:
            tot += cnt[end]
            end += 1
        c = cnt[start:end]
        total = int(c.sum())
        base = np.repeat(lo[start:end], c)
        idx = base + (np.arange(total) - np.repeat(np.cumsum(c) - c, c))
        acc_l.append(np.repeat(lrow[start:end], c))
        acc_r.append(inv_rows[idx])
        acc_t += total
        start = end
    L = np.concatenate(acc_l).astype(np.int64)
    R = np.concatenate(acc_r).astype(np.int64)
    if stats is not None:
        stats[bit] = acc_t
    return L, R


def ragged_range(starts, lens):
    """Concatenation of arange(s, s+l) for every (s, l) pair -- vectorized."""
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    out = np.ones(total, np.int64)
    out[0] = starts[0]
    if len(lens) > 1:
        ends = np.cumsum(lens)[:-1]
        out[ends] = starts[1:] - (starts[:-1] + lens[:-1]) + 1
    return np.cumsum(out)


def filter_csr(off, keys, keep):
    """Restrict a CSR to the rows selected by a boolean mask."""
    lens = np.diff(off)[keep]
    new_off = np.zeros(int(keep.sum()) + 1, np.int64)
    new_off[1:] = np.cumsum(lens)
    starts = off[:-1][keep].astype(np.int64)
    return new_off, keys[ragged_range(starts, lens.astype(np.int64))]


def extra_keys(present):
    """(nsuf, askel) CSRs of a Source, or None when 02b_extra_keys.py has not run."""
    out = []
    for f in ("nsuf", "askel"):
        p = os.path.join(common.PREP, f"{present.tag}_{f}")
        if not (os.path.exists(p + ".npy") and os.path.exists(p + ".off.npy")):
            return None
        out.append((np.load(p + ".off.npy", mmap_mode="r"),
                    np.load(p + ".npy", mmap_mode="r")))
    return out


def run_split(split, left_sample=0, recall=False, caps=None, out_tag=None,
              fams=None):
    caps = caps or DEFAULT_CAPS
    t0 = time.time()
    left = Source(f"{split}_s1")
    right = Right(split)
    full_left = len(left)
    n_right = right.n
    fams = fams or list(DEFAULT_CAPS)
    print(f"[{split}] left={full_left} right={n_right} fams={fams}", flush=True)

    # ---- optional left subsample ---------------------------------------
    left_idx = None
    if left_sample and left_sample < full_left:
        left_idx = np.sort(np.random.default_rng(0).choice(
            full_left, left_sample, replace=False))
        keep = np.zeros(full_left, bool)
        keep[left_idx] = True
        l_ntok_off, l_ntok = filter_csr(left.ntok_off, np.asarray(left.ntok), keep)
        l_npre_off, l_npre = filter_csr(left.npre_off, np.asarray(left.npre), keep)
        l_atok_off, l_atok = filter_csr(left.atok_off, np.asarray(left.atok), keep)
        ex = extra_keys(left)
        l_extra = [filter_csr(o, np.asarray(k), keep) for o, k in ex] if ex else None
        l_nset = set_key(l_ntok, l_ntok_off, 0x5DEECE66D)
        l_aset = set_key(l_atok, l_atok_off, 0x2545F491)
        l_hnum = num_key(np.asarray(left.hnum)[keep], 0x1234567,
                         np.asarray(left.hnum)[keep] > 0)
        l_zip = num_key(np.asarray(left.zpc)[keep], 0x7654321,
                        np.asarray(left.zpc)[keep] > 0)
        n_left = left_sample
    else:
        l_ntok_off, l_ntok = left.ntok_off, np.asarray(left.ntok)
        l_npre_off, l_npre = left.npre_off, np.asarray(left.npre)
        l_atok_off, l_atok = left.atok_off, np.asarray(left.atok)
        ex_l = extra_keys(left)
        l_extra = ex_l if ex_l else None
        l_nset = set_key(l_ntok, l_ntok_off, 0x5DEECE66D)
        l_aset = set_key(l_atok, l_atok_off, 0x2545F491)
        l_hnum = num_key(np.asarray(left.hnum), 0x1234567, np.asarray(left.hnum) > 0)
        l_zip = num_key(np.asarray(left.zpc), 0x7654321, np.asarray(left.zpc) > 0)
        n_left = full_left
    one = np.arange(n_left + 1, dtype=np.int32)

    families = {
        "name_token": (l_ntok_off, l_ntok),
        "name_prefix4": (l_npre_off, l_npre),
        "name_set": (one, l_nset.astype(np.int32)),
        "addr_token": (l_atok_off, l_atok),
        "addr_set": (one, l_aset.astype(np.int32)),
        "house_number": (one, l_hnum.astype(np.int32)),
        "zip": (one, l_zip.astype(np.int32)),
    }
    r_ntok = np.asarray(right.ntok)
    r_npre = np.asarray(right.npre)
    r_atok = np.asarray(right.atok)
    ex_r = extra_keys(right.s2)
    ex_r3 = extra_keys(right.s3) if ex_r else None
    if ex_r and ex_r3 and l_extra is not None:
        cat_csr = lambda o1, t1, o2, t2: (
            np.concatenate([o1, o1[-1] + o2[1:]]).astype(np.int32),
            np.concatenate([np.asarray(t1), np.asarray(t2)]))
        r_extra = {nm: cat_csr(ex_r[i][0], ex_r[i][1], ex_r3[i][0], ex_r3[i][1])
                   for i, nm in enumerate(("nsuf", "askel"))}
    else:
        r_extra = {}
    right_side = {
        "name_token": (right.ntok_off, r_ntok),
        "name_prefix4": (right.npre_off, r_npre),
        "name_set": (None, set_key(r_ntok, right.ntok_off, 0x5DEECE66D)),
        "addr_token": (right.atok_off, r_atok),
        "addr_set": (None, set_key(r_atok, right.atok_off, 0x2545F491)),
        "house_number": (None, num_key(right.hnum, 0x1234567,
                                       np.asarray(right.hnum) > 0)),
        "zip": (None, num_key(right.zpc, 0x7654321, np.asarray(right.zpc) > 0)),
    }
    for i, nm in enumerate(("nsuf", "askel")):
        if nm in r_extra:
            right_side[nm] = r_extra[nm]
            families[nm] = l_extra[i]
    missing = [f for f in fams if f not in families]
    if missing:
        raise SystemExit(f"families not available: {missing}")
    fams = [f for f in DEFAULT_CAPS if f in fams]

    all_l, all_r, all_m = [], [], []
    for bit, name in enumerate(fams, start=1):
        loff, lkeys = families[name]
        roff, rkeys = right_side[name]
        if roff is None:
            roff = np.arange(n_right + 1, dtype=np.int32)
            rkeys = rkeys.astype(np.int32)
        else:
            rkeys = np.asarray(rkeys)
        inv_keys, inv_rows = build_index(rkeys, roff, n_right)
        kv, kc = key_counts(rkeys)
        res = expand(loff, lkeys, inv_keys, inv_rows, kv, kc, caps[name], bit)
        if res is None:
            print(f"   {name:<14s} cap={caps[name]:<4d} -> 0 pairs", flush=True)
            continue
        L, R = res
        print(f"   {name:<14s} cap={caps[name]:<4d} -> {len(L):>12,d} pairs "
              f"[{time.time()-t0:.0f}s]", flush=True)
        all_l.append(L)
        all_r.append(R)
        all_m.append(np.full(len(L), bit, np.uint8))
        del inv_keys, inv_rows, kv, kc

    L = np.concatenate(all_l)
    R = np.concatenate(all_r)
    M = np.concatenate(all_m)
    del all_l, all_r, all_m, families, right_side
    print(f"   raw pairs: {len(L):,d}  [{time.time()-t0:.0f}s]", flush=True)

    packed = (L << 24) | R
    order = np.argsort(packed, kind="stable")
    packed = packed[order]
    msorted = M[order]
    del order, L, R, M
    first = np.empty(len(packed), bool)
    first[0] = True
    np.not_equal(packed[1:], packed[:-1], out=first[1:])
    grp_starts = np.flatnonzero(first)
    mask = np.bitwise_or.reduceat(msorted, grp_starts).astype(np.uint8)
    upacked = packed[grp_starts]
    del packed, msorted
    ul = (upacked >> 24).astype(np.int32)
    ur = (upacked & MASK).astype(np.int32)
    print(f"   unique pairs: {len(ul):,d}  [{time.time()-t0:.0f}s]", flush=True)

    tag = out_tag or split
    np.save(os.path.join(WORK, f"cand_{tag}_l.npy"), ul)
    np.save(os.path.join(WORK, f"cand_{tag}_r.npy"), ur)
    np.save(os.path.join(WORK, f"cand_{tag}_m.npy"), mask)
    with open(os.path.join(WORK, f"cand_{tag}_meta.txt"), "w") as f:
        f.write(f"n_left={n_left}\nn_right={n_right}\nn2={right.n2}\n")
    if left_idx is not None:
        np.save(os.path.join(WORK, f"cand_{tag}_leftidx.npy"), left_idx)

    if recall:
        eval_recall(split, ul, ur, n_left, n_right, left_idx)
    print(f"[{split}] blocking done in {time.time()-t0:.0f}s", flush=True)
    return ul, ur, mask


def _filter_off(off, keep):
    cnt = np.diff(off) * keep
    out = np.zeros(off.shape[0], np.int64)
    out[1:] = np.cumsum(cnt)
    return out.astype(np.int64)


def eval_recall(split, ul, ur, n_left, n_right, left_idx=None):
    """Fraction of ground-truth pairs that blocking retrieved."""
    if split != "train":
        return
    tl, tr = np.load(os.path.join(WORK, "gt_s1.npy")), np.load(os.path.join(WORK, "gt_right.npy"))
    if left_idx is not None:
        pos = np.full(len(Source("train_s1")), -1, np.int64)
        pos[left_idx] = np.arange(len(left_idx))
        keep = pos[tl] >= 0
        tl = pos[tl[keep]]
        tr = tr[keep]
        ntot = int(keep.sum())
    else:
        ntot = len(tl)
    pred = np.unique(ul.astype(np.int64) << 24 | ur)
    true = np.unique(tl.astype(np.int64) << 24 | tr)
    hit = np.intersect1d(pred, true, assume_unique=True)
    rec = len(hit) / max(ntot, 1)
    print(f"   BLOCKING RECALL: {rec:.4f}  ({len(hit):,d} / {ntot:,d} true pairs found)",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample-left", type=int, default=0)
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fams", default="",
                    help="comma separated subset of name_token,name_prefix4,name_set,"
                         "addr_token,addr_set,house_number,zip,nsuf,askel")
    ap.add_argument("--caps", default="", help="k=v,... overrides for the probe")
    args = ap.parse_args()
    caps = dict(DEFAULT_CAPS)
    for kv in filter(None, args.caps.split(",")):
        k, v = kv.split("=")
        caps[k] = int(v)
    run_split(args.split, left_sample=args.sample_left, recall=args.recall,
              out_tag=args.tag or None,
              caps=caps if args.caps else None,
              fams=[f for f in filter(None, args.fams.split(","))] or None)


if __name__ == "__main__":
    main()
