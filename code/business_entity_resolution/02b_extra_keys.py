"""
Extra blocking keys: name suffix-4 (``nsuf``) and address skeleton token (``askel``).

The six families used in production leave 15.1% of the true pairs unretrieved
(see exp_miss2.py: over 30k actually-missed pairs, a shared name suffix-4 key
recovers 81% of them and a shared address skeleton token 86%, and the union of
the two 98.7%).  Both keys are derived from the canonical strings that
01_preprocess.py already wrote, so this pass only re-reads the blobs:

    <tag>_nsuf.npy / <tag>_nsuf.off.npy   int32  last 4 chars of each name token
    <tag>_askel.npy / <tag>_askel.off.npy int32  vowel-stripped, digit-folded addr token

Key ids use the same 25-bit blake2b hashing trick as the existing token ids,
with a distinct salt so they cannot collide with npre/atok postings.

Usage:  python 02b_extra_keys.py [--jobs 10] [--tags train_s1,...]
"""
import argparse
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erlib as E  # noqa: E402
from common import PREP  # noqa: E402

HASH_MASK = (1 << 25) - 1
_b2b = __import__("hashlib").blake2b
DIGIT2LET = str.maketrans({"0": "o", "1": "l", "2": "z", "3": "e", "4": "a",
                           "5": "s", "6": "b", "7": "t", "8": "b", "9": "g"})
VOW = re.compile(r"[aeiou]")
CHUNK = 200_000


def key_id(salt, tok):
    return int.from_bytes(_b2b(salt + tok.encode("ascii", "ignore"), digest_size=8).digest(),
                          "little") & HASH_MASK


def suffix_ids(core):
    seen, out = set(), []
    for t in core.split():
        i = key_id(b"s4", t[-4:])
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def skel_ids(core):
    seen, out = set(), []
    for t in core.split():
        if t == "unit":
            continue
        s = VOW.sub("", t.translate(DIGIT2LET))[:6]
        if not s:
            continue
        i = key_id(b"ak", s)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def read_span(blob_path, off, start, end):
    with open(blob_path, "rb") as f:
        f.seek(int(off[start]))
        return f.read(int(off[end]) - int(off[start]))


def split_rows(buf, lens):
    out, p = [], 0
    for L in lens:
        out.append(buf[p:p + L])
        p += L
    return out


def work(task):
    tag, start, end = task
    noff = np.load(os.path.join(PREP, f"{tag}_name.off.npy"), mmap_mode="r")
    aoff = np.load(os.path.join(PREP, f"{tag}_addr.off.npy"), mmap_mode="r")
    nlen = (noff[start + 1:end + 1] - noff[start:end]).astype(np.int64)
    alen = (aoff[start + 1:end + 1] - aoff[start:end]).astype(np.int64)
    nbuf = read_span(os.path.join(PREP, f"{tag}_name.blob"), noff, start, end)
    abuf = read_span(os.path.join(PREP, f"{tag}_addr.blob"), aoff, start, end)
    names = split_rows(nbuf, nlen)
    addrs = split_rows(abuf, alen)
    ns, ak, n_cnt, a_cnt = [], [], [], []
    for nb, ab in zip(names, addrs):
        ns.append(suffix_ids(E.norm_name(nb.decode("utf-8", "replace"))[1]))
        ak.append(skel_ids(E.norm_addr(ab.decode("utf-8", "replace"), "")[1]))
        n_cnt.append(len(ns[-1]))
        a_cnt.append(len(ak[-1]))
    n_cnt = np.array(n_cnt, np.int32)
    a_cnt = np.array(a_cnt, np.int32)
    n_off = np.zeros(len(n_cnt) + 1, np.int32)
    np.cumsum(n_cnt, out=n_off[1:], dtype=np.int32)
    a_off = np.zeros(len(a_cnt) + 1, np.int32)
    np.cumsum(a_cnt, out=a_off[1:], dtype=np.int32)
    n_flat = np.fromiter((x for t in ns for x in t), np.int32, count=int(n_off[-1]))
    a_flat = np.fromiter((x for t in ak for x in t), np.int32, count=int(a_off[-1]))
    return n_flat, n_off, a_flat, a_off


def rebase(parts, dtype):
    out = np.zeros(sum(len(a) for a in parts) + 1, dtype)
    base = dtype(0)
    pos = 0
    for a in parts:
        out[pos + 1: pos + 1 + len(a)] = a + base
        pos += len(a)
        base = base + dtype(a[-1]) if len(a) else base
    return out


def build(tag, jobs):
    t = time.time()
    n = len(np.load(os.path.join(PREP, f"{tag}_id.npy"), mmap_mode="r"))
    tasks = [(tag, s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]
    nf, no, af, ao = [], [], [], []
    if jobs > 1:
        with Pool(jobs) as pool:
            wave = jobs * 2
            for i in range(0, len(tasks), wave):
                for nfl, nof, afl, aof in pool.map(work, tasks[i:i + wave]):
                    nf.append(nfl)
                    no.append(np.asarray(nof[1:], np.int32))
                    af.append(afl)
                    ao.append(np.asarray(aof[1:], np.int32))
                print(f"  {tag}: {min(i+wave, len(tasks))}/{len(tasks)} chunks "
                      f"[{time.time()-t:.0f}s]", flush=True)
    else:
        for tsk in tasks:
            nfl, nof, afl, aof = work(tsk)
            nf.append(nfl)
            no.append(np.asarray(nof[1:], np.int32))
            af.append(afl)
            ao.append(np.asarray(aof[1:], np.int32))
    noff = rebase(no, np.int32)
    aoff = rebase(ao, np.int32)
    np.save(os.path.join(PREP, f"{tag}_nsuf.npy"),
            np.concatenate(nf) if nf else np.zeros(0, np.int32))
    np.save(os.path.join(PREP, f"{tag}_nsuf.off.npy"), noff)
    np.save(os.path.join(PREP, f"{tag}_askel.npy"),
            np.concatenate(af) if af else np.zeros(0, np.int32))
    np.save(os.path.join(PREP, f"{tag}_askel.off.npy"), aoff)
    print(f"{tag}: rows={n:,} nnz_nsuf={int(noff[-1]):,} nnz_askel={int(aoff[-1]):,} "
          f"[{time.time()-t:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 4))
    ap.add_argument("--tags", default="train_s1,train_s2,train_s3,"
                                      "test_s1,test_s2,test_s3")
    args = ap.parse_args()
    for tag in args.tags.split(","):
        build(tag, args.jobs)


if __name__ == "__main__":
    main()
