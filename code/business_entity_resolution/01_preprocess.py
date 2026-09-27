"""
Stage 1 -- preprocessing / canonicalization of the raw TSV sources.

For every (split, source) pair this writes a compact, memory-mappable
representation to ``work/prep``:

    <tag>_id.npy      int64   numeric part of entity_id  ("S1-925783039" -> 925783039)
    <tag>_ctry.npy    uint8   country code (open set: unknown -> 0)
    <tag>_flag.npy    uint8   bit0 name had a non-Latin script
                              bit1 address had a non-Latin script
                              bit2 address is empty
    <tag>_name.off    uint64  (n+1) byte offsets into <tag>_name.blob
    <tag>_name.blob   bytes   canonical business_name (NFKD-folded, lower, symbol-folded)
    <tag>_addr.off    uint64  (n+1) byte offsets into <tag>_addr.blob
    <tag>_addr.blob   bytes   canonical business_address (abbreviations + states folded)
    <tag>_ntok.off    int32   (n+1) CSR row pointers into <tag>_ntok.npy
    <tag>_ntok.npy    int32   business_name token ids (hashing trick, 2**25 space)
    <tag>_atok.off    int32   (n+1) CSR row pointers into <tag>_atok.npy
    <tag>_atok.npy    int32   business_address token ids (2**25 space)

Token ids come from the hashing trick (blake2b -> 25 bits).  That removes the
need for a global token dictionary, keeps the postings in a plain int32 array
and still gives exact document frequencies (hence exact IDF) via bincount.
Collisions between distinct tokens are rare and only merge two rare blocking
keys, which costs recall nothing and precision nothing (the model re-scores
every candidate pair from the raw strings anyway).

Usage:  python 01_preprocess.py [--jobs 10] [--limit 0] [--tag-suffix ""]
"""

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erlib as E  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "student_resource", "dataset")
PREP = os.path.join(ROOT, "work", "prep")

HASH_BITS = 25
HASH_MASK = (1 << HASH_BITS) - 1

_b2b = __import__("hashlib").blake2b


def tok_id(tok: str) -> int:
    return int.from_bytes(_b2b(tok.encode("ascii", "ignore"), digest_size=8).digest(),
                          "little") & HASH_MASK


def pre_id(tok: str) -> int:
    """Blocking key for a token: 4-char prefix, so that single-character typos
    ("morrow" / "morrw", "fashion" / "fashi0n") still land on one key."""
    t = tok[:4] if len(tok) > 4 else tok
    return int.from_bytes(_b2b(b"p4" + t.encode("ascii", "ignore"), digest_size=8).digest(),
                          "little") & HASH_MASK


def token_ids(text: str) -> list:
    if not text:
        return []
    seen = set()
    out = []
    for t in text.split():
        i = tok_id(t)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def prefix_ids(text: str) -> list:
    if not text:
        return []
    seen = set()
    out = []
    for t in text.split():
        i = pre_id(t)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def address_numbers(addr_norm: str):
    """(house number, postal code) of a canonical address."""
    hnum = 0
    zpc = 0
    for t in addr_norm.split():
        if t.isdigit():
            if not hnum:
                hnum = int(t)
            if 4 <= len(t) <= 6:
                zpc = int(t)
    return hnum, zpc


def process_chunk(args):
    """Worker: read a slice of a source TSV and return canonical arrays."""
    path, start, nrows, ctry_map = args
    skiprows = 1 + start
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, skiprows=skiprows,
        nrows=nrows, header=None, names=["entity_id", "business_name",
                                         "business_address", "country"],
        engine="c", quoting=3,
    )
    eid = df.entity_id.to_numpy()
    name = df.business_name.to_numpy()
    addr = df.business_address.to_numpy()
    ctry = df.country.to_numpy()

    n = len(df)
    ids = np.empty(n, dtype=np.int64)
    names = []
    ab_addrs = []
    nb_len = np.empty(n, dtype=np.int64)
    ab_len = np.empty(n, dtype=np.int64)
    n_cnt = np.empty(n, dtype=np.int32)
    a_cnt = np.empty(n, dtype=np.int32)
    p_cnt = np.empty(n, dtype=np.int32)
    hnum = np.zeros(n, dtype=np.int32)
    zpc = np.zeros(n, dtype=np.int32)
    flag = np.zeros(n, dtype=np.uint8)
    cc = np.empty(n, dtype=np.uint8)

    ntok, atok, ptok = [], [], []
    for i in range(n):
        ids[i] = int(eid[i][3:])
        c = ctry[i].strip().lower()
        code = ctry_map.get(c, 0)
        cc[i] = code
        nn, nc, nl = E.norm_name(name[i])
        an, ac, al = E.norm_addr(addr[i], ctry[i])
        b = nn.encode("utf-8")
        ab = an.encode("utf-8")
        names.append(b)
        ab_addrs.append(ab)
        nb_len[i] = len(b)
        ab_len[i] = len(ab)
        t = token_ids(nc)
        a = token_ids(ac)
        p = prefix_ids(nc)
        n_cnt[i] = len(t)
        a_cnt[i] = len(a)
        p_cnt[i] = len(p)
        ntok.append(t)
        atok.append(a)
        ptok.append(p)
        hn, zp = address_numbers(an)
        hnum[i] = hn % (2 ** 31 - 1)
        zpc[i] = zp % (2 ** 31 - 1)
        f = 0
        if nl:
            f |= 1
        if al:
            f |= 2
        if not an:
            f |= 4
        flag[i] = f

    noff = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(nb_len, out=noff[1:])
    aoff_s = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(ab_len, out=aoff_s[1:])
    toff = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(n_cnt, out=toff[1:], dtype=np.int32)
    aoff = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(a_cnt, out=aoff[1:], dtype=np.int32)
    poff = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(p_cnt, out=poff[1:], dtype=np.int32)

    ntok_flat = np.fromiter((x for t in ntok for x in t), dtype=np.int32,
                            count=int(toff[-1]))
    atok_flat = np.fromiter((x for t in atok for x in t), dtype=np.int32,
                            count=int(aoff[-1]))
    ptk_flat = np.fromiter((x for t in ptok for x in t), dtype=np.int32,
                            count=int(poff[-1]))
    return (ids, cc, flag, names, noff, toff, ntok_flat, aoff, atok_flat,
            ptk_flat, hnum, zpc, poff, ab_addrs, aoff_s)



def build_country_table(splits, sources):
    codes = {}
    for split in splits:
        for s in sources:
            p = source_path(split, s)
            out = subprocess.run(["cut", "-f4", p], capture_output=True, text=True)
            vals = out.stdout.split("\n")[1:]          # drop header
            for v in set(vals):
                key = v.strip().lower()
                if key and key not in codes:
                    codes[key] = len(codes) + 1
    return {"": 0, **{k: v for k, v in codes.items()}}, codes


def source_path(split, s):
    name = "train" if split == "train" else "test"
    return os.path.join(DATA, split, f"{name}_source{s}.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0, help="rows per source (debug)")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--sources", default="1,2,3")
    args = ap.parse_args()

    os.makedirs(PREP, exist_ok=True)
    t0 = time.time()
    splits = args.splits.split(",")
    sources = [int(x) for x in args.sources.split(",")]
    ctry_map, codes = build_country_table(splits, sources)
    with open(os.path.join(PREP, "country_table.txt"), "w") as f:
        for k, v in sorted(codes.items(), key=lambda kv: kv[1]):
            f.write(f"{v}\t{k}\n")
    print(f"country table ({len(codes)}): "
          f"{ {c: k for k, c in codes.items()} }  [{time.time()-t0:.0f}s]")

    for split in splits:
        for s in sources:
            tag = f"{split}_s{s}"
            path = source_path(split, s)
            total = count_rows(path)
            if args.limit:
                total = min(total, args.limit)
            build_one(path, tag, total, ctry_map, args.jobs)
    print(f"preprocessing done in {time.time()-t0:.0f}s")


def count_rows(path):
    out = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) - 1


def _rebase(parts, dtype):
    """Turn chunk-local cumulative offsets into one global indptr array."""
    out = np.zeros(sum(len(a) for a in parts) + 1, dtype)
    base = dtype(0)
    pos = 0
    for a in parts:
        out[pos + 1: pos + 1 + len(a)] = a + base
        pos += len(a)
        base = base + dtype(a[-1]) if len(a) else base
    return out


def build_one(path, tag, total, ctry_map, jobs):
    t = time.time()
    nrows = 150_000
    starts = list(range(0, total, nrows))
    tasks = [(path, st, min(nrows, total - st), ctry_map) for st in starts]

    id_parts, cc_parts, fl_parts, hn_parts, zp_parts = [], [], [], [], []
    name_parts, addr_parts, ntok_parts, atok_parts, ptk_parts = [], [], [], [], []
    # each entry is a chunk-local cumulative array; the base of the next chunk
    # is only known once the previous ones are seen, so fix the bases up at the end
    noffs, toffs, aoffs, poffs, aoffs_s = [], [], [], [], []

    def handle(res):
        (ids, cc, flag, names, noff, toff, nt, aoff, at, pt, hn, zp, poff,
         ab_addrs, aoff_s) = res
        id_parts.append(ids)
        cc_parts.append(cc)
        fl_parts.append(flag)
        hn_parts.append(hn)
        zp_parts.append(zp)
        name_parts.extend(names)
        addr_parts.extend(ab_addrs)
        ntok_parts.append(nt)
        atok_parts.append(at)
        ptk_parts.append(pt)
        noffs.append(np.asarray(noff[1:], np.int64))
        aoffs_s.append(np.asarray(aoff_s[1:], np.int64))
        toffs.append(np.asarray(toff[1:], np.int32))
        aoffs.append(np.asarray(aoff[1:], np.int32))
        poffs.append(np.asarray(poff[1:], np.int32))

    if jobs > 1:
        with Pool(jobs) as pool:
            wave = jobs * 2
            for i in range(0, len(tasks), wave):
                async_res = [pool.apply_async(process_chunk, (t_,)) for t_ in tasks[i:i + wave]]
                for ar in async_res:
                    handle(ar.get())
                print(f"  {tag}: {min(i+wave, len(tasks))}/{len(tasks)} chunks "
                      f"[{time.time()-t:.0f}s]", flush=True)
    else:
        for t_ in tasks:
            handle(process_chunk(t_))

    ids = np.concatenate(id_parts)
    cc = np.concatenate(cc_parts)
    flag = np.concatenate(fl_parts)
    noff = _rebase(noffs, np.int64)
    aoff_s = _rebase(aoffs_s, np.int64)
    toff = _rebase(toffs, np.int32)
    aoff = _rebase(aoffs, np.int32)
    poff = _rebase(poffs, np.int32)
    hnum = np.concatenate(hn_parts)
    zpc = np.concatenate(zp_parts)
    ntok = np.concatenate(ntok_parts) if ntok_parts else np.zeros(0, np.int32)
    atok = np.concatenate(atok_parts) if atok_parts else np.zeros(0, np.int32)
    ptk = np.concatenate(ptk_parts) if ptk_parts else np.zeros(0, np.int32)

    np.save(os.path.join(PREP, f"{tag}_id.npy"), ids)
    np.save(os.path.join(PREP, f"{tag}_ctry.npy"), cc)
    np.save(os.path.join(PREP, f"{tag}_flag.npy"), flag)
    np.save(os.path.join(PREP, f"{tag}_ntok.off.npy"), toff)
    np.save(os.path.join(PREP, f"{tag}_ntok.npy"), ntok)
    np.save(os.path.join(PREP, f"{tag}_atok.off.npy"), aoff)
    np.save(os.path.join(PREP, f"{tag}_atok.npy"), atok)
    np.save(os.path.join(PREP, f"{tag}_npre.off.npy"), poff)
    np.save(os.path.join(PREP, f"{tag}_npre.npy"), ptk)
    np.save(os.path.join(PREP, f"{tag}_hnum.npy"), hnum)
    np.save(os.path.join(PREP, f"{tag}_zpc.npy"), zpc)
    with open(os.path.join(PREP, f"{tag}_name.blob"), "wb") as f:
        for b in name_parts:
            f.write(b)
    np.save(os.path.join(PREP, f"{tag}_name.off.npy"), noff)
    with open(os.path.join(PREP, f"{tag}_addr.blob"), "wb") as f:
        for b in addr_parts:
            f.write(b)
    np.save(os.path.join(PREP, f"{tag}_addr.off.npy"), aoff_s)

    print(f"{tag}: rows={len(ids)} nnz_name={len(ntok)} nnz_pre={len(ptk)} "
          f"nnz_addr={len(atok)} "
          f"[{time.time()-t:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
