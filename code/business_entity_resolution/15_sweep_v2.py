"""Pick the operating point for the stage-2b ranker, honestly.

Three pools can contribute pairs (base blocking, the targeted recovery pass, the
conjunctive pass) and the ranker changed, so the v1 operating point no longer
applies.  Each pool gets its own threshold, fitted by coordinate ascent on the
full ground truth of the held-out entity split:

    base pool  >= t_base
    recovery   >= t_rec
    conjunctive>= t_conj

Fitting three scalars on 220,682 entities is a negligible fit, and the point of
this script is that the number it reports comes from the full ground truth --
including the entities that no pool ever retrieved, which is where the earlier
0.95393 figure went wrong.

Usage:
  python 15_sweep_v2.py --score sv2
  python 15_sweep_v2.py --score sv            # the v1 comparison
"""
import argparse
import importlib
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M6 = importlib.import_module("06_model")
import common
from common import WORK, Right, Source

POOLS = (("", "base"), ("_recv2", "rec"), ("_conj", "conj"))


def load_pool(split, tag, name, is_valid, nf):
    """Load one candidate pool and its scores.  `name` selects the score set,
    which is what lets a pool be scored by whichever ranker suits it: the base
    pool wants stage-2b, the recall pools want the model they were tuned with,
    because stage-2b has only ever seen base-pool rows in training."""
    p = os.path.join(WORK, f"pairs_{split}{tag}_")
    l = np.asarray(np.load(p + "l.npy")).astype(np.int64)
    r = np.asarray(np.load(p + "r.npy")).astype(np.int64)
    y = np.asarray(np.load(p + "y.npy"), np.float64)
    sp = p + f"{name}.npy"
    if os.path.exists(sp):
        s = np.asarray(np.load(sp), np.float64)
    elif tag == "" and name == "sv":
        s = None
    else:
        mdl = stage2b_model(split)
        d = {"l": l, "f": np.load(p + "f.npy", mmap_mode="r"),
             "fz": np.load(p + "fz.npy", mmap_mode="r")}
        s = np.empty(len(l), np.float64)
        for b0 in range(0, len(l), 2_000_000):
            b1 = min(b0 + 2_000_000, len(l))
            s[b0:b1] = mdl.predict_proba(M6.gather(d, np.arange(b0, b1), nf))[:, 1]
        np.save(sp, s.astype(np.float32))
    va = is_valid[l]
    if s is not None and len(s) != len(l):
        # pairs_train_sv.npy holds only the held-out rows, in va_idx order
        full = np.empty(len(l), np.float64)
        full[va] = s
        s = full
    return l[va], r[va], y[va], s[va]


def stage2b_model(split):
    with open(os.path.join(WORK, f"model2b_{split}.pkl"), "rb") as f:
        return pickle.load(f)["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="sv2")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--mix", default="", help="base=sv2,rec=sv,conj=sv")
    ap.add_argument("--pools", default="base,rec,conj",
                    help="comma-separated subset of base,rec,conj to include")
    args = ap.parse_args()
    split = "train"
    L, R = Source(f"{split}_s1"), Right(split)
    n_all = len(L)
    nf = np.load(os.path.join(WORK, f"pairs_{split}_f.npy"), mmap_mode="r").shape[1]
    rng = np.random.default_rng(M6.SEED)
    is_valid = np.zeros(n_all, bool)
    is_valid[rng.choice(n_all, int(n_all * M6.VALID_FRAC), replace=False)] = True
    ents = np.flatnonzero(is_valid)
    remap = np.full(n_all, -1, np.int64)
    remap[ents] = np.arange(len(ents))
    n_val = len(ents)

    use_names = [x.strip() for x in args.pools.split(",") if x.strip()]
    score_of = dict(x.split("=") for x in args.mix.split(",")) if args.mix else {}
    pl_all, pr_all, ps, py, pid = [], [], [], [], []
    for tag, name in POOLS:
        if name not in use_names:
            continue
        l, r, y, s = load_pool(split, tag, score_of.get(name, args.score), is_valid, nf)
        print(f"pool {name:<6s}({score_of.get(name, args.score):>4s})"
              f"{len(l):>10,} pairs {int(y.sum()):>9,} positives "
              f"mean score {s.mean():.4f}", flush=True)
        pl_all.append(remap[l])
        pr_all.append(r)
        ps.append(s)
        py.append(y)
        pid.append(np.full(len(l), len(pid), np.int8))
    cl = np.concatenate(pl_all)
    cr = np.concatenate(pr_all)
    ps = np.concatenate(ps)
    py = np.concatenate(py)
    pid = np.concatenate(pid)
    assert len(cl) == len(cr) == len(ps) == len(py) == len(pid), (
        f"pool arrays disagree: {len(cl)} {len(cr)} {len(ps)} {len(py)} {len(pid)}")
    assert R.n <= (1 << 24), "the dedup key would collide"
    key = (cl << 24) | cr
    _, idx = np.unique(key, return_index=True)
    idx = np.sort(idx)
    cl, cr, py, ps, pid = cl[idx], cr[idx], py[idx], ps[idx], pid[idx]
    print(f"merged {len(cl):,} unique pairs, {int(py.sum()):,} positives", flush=True)

    tl, tr = common.load_gt("train")
    mm = is_valid[tl]
    tl, tr = remap[tl[mm]], tr[mm]
    ntrue = np.bincount(tl, minlength=n_val).astype(np.int64)

    o = np.argsort(-ps, kind="stable")
    cl, cr, py, ps, pid = cl[o], cr[o], py[o], ps[o], pid[o]

    def macro(thrs):
        use = np.zeros(len(ps), bool)
        for i, t in enumerate(thrs):
            use |= (pid == i) & (ps >= t)
        npred = np.bincount(cl[use], minlength=n_val).astype(np.int64)
        tp = np.bincount(cl[use], weights=py[use], minlength=n_val)
        return float(common.entity_fbeta(tp, npred, ntrue).mean()), use

    grid = np.unique(np.round(np.concatenate([
        np.linspace(0.30, 0.99, 70), 1.0 - np.geomspace(1e-6, 0.5, 40)]), 6))
    thrs = {"base": 0.669231, "rec": 0.794601, "conj": 0.783333}
    thrs = [thrs[x[1]] for x in POOLS if x[1] in use_names]
    best, _ = macro(thrs)
    print(f"start {np.round(thrs,6).tolist()} -> {best:.5f}", flush=True)
    for rnd in range(args.iters):
        improved = False
        for i in range(len(thrs)):
            cands = grid if rnd == 0 else np.unique(
                np.round(np.clip(np.linspace(thrs[i] - 0.06, thrs[i] + 0.06, 25),
                                 0.0, 1.0), 6))
            for t in cands:
                trial = list(thrs)
                trial[i] = float(t)
                m, _ = macro(trial)
                if m > best + 1e-7:
                    best, thrs, improved = m, trial, True
        print(f"round {rnd}: {np.round(thrs, 6).tolist()} -> {best:.5f}", flush=True)
        if not improved:
            break

    m, use = macro(thrs)
    npred = np.bincount(cl[use], minlength=n_val).astype(np.int64)
    tp = np.bincount(cl[use], weights=py[use], minlength=n_val)
    print(f"\nfinal macro={m:.5f}  pred={int(npred.sum()):,} "
          f"tp={int(round(tp.sum())):,}  precision={tp.sum()/max(npred.sum(),1):.4f}")
    print(common.fmt("MERGED", common.score(cl[use], cr[use], tl, tr, n_val, R.n)))
    suffix = (args.mix.replace("=", "-").replace(",", "+") or args.score) + "_" \
        + args.pools.replace(",", "-")
    with open(os.path.join(WORK, f"sweep_{suffix}.txt"), "w") as f:
        for (tag, name), v in zip([x for x in POOLS if x[1] in use_names], thrs):
            f.write(f"{name}={v}\n")
        f.write(f"macro={m}\nscore={args.score}\npools={args.pools}\n")
    tag_out = suffix
    np.save(os.path.join(WORK, f"merged_{tag_out}_sel.npy"), use)
    np.save(os.path.join(WORK, f"merged_{tag_out}_l.npy"), cl)
    np.save(os.path.join(WORK, f"merged_{tag_out}_r.npy"), cr)
    print(f"-> work/sweep_{suffix}.txt")


if __name__ == "__main__":
    main()
