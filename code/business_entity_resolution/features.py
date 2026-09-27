"""Vectorized pair features: sparse TF-IDF cosine + token/numeric overlap."""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HASH_BITS = 25
HASH_SIZE = 1 << HASH_BITS


def tfidf_matrix(tokens, indptr, n_rows, idf_pow=1.0, binary=True, sublinear=True):
    """CSR of L2-normalized tf-idf weights; columns are the raw hashed token ids."""
    tokens = np.asarray(tokens)
    indptr = np.asarray(indptr)
    df = np.bincount(tokens, minlength=HASH_SIZE)
    n = n_rows
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    del df
    idf = idf.astype(np.float32)
    data = idf[tokens]
    if sublinear:
        pass
    if idf_pow != 1.0:
        idf = idf ** idf_pow
        data = idf[tokens]
    X = sp.csr_matrix((data.astype(np.float32), tokens.astype(np.int32),
                       indptr.astype(np.int64)), shape=(n_rows, HASH_SIZE))
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    X.data /= np.repeat(norms.astype(np.float32), np.diff(indptr))
    return X


def row_nnz(indptr):
    return np.diff(np.asarray(indptr)).astype(np.int32)


def jaccard(shared, na, nb):
    s = shared.astype(np.float32)
    uni = np.maximum(na + nb - shared, 1).astype(np.float32)
    return s / uni, s / np.maximum(np.minimum(na, nb), 1), s / np.maximum(np.maximum(na, nb), 1)


def build_features(L, R, ln, rn, mats=None):
    """Full cheap-feature bundle for a chunk of pairs."""
    ln = np.asarray(ln, dtype=np.int64)
    rn = np.asarray(rn, dtype=np.int64)
    if mats is None:
        mats = {
            "n": (tfidf_matrix(L.ntok, L.ntok_off, len(L)),
                  tfidf_matrix(R.ntok, R.ntok_off, R.n)),
            "a": (tfidf_matrix(L.atok, L.atok_off, len(L)),
                  tfidf_matrix(R.atok, R.atok_off, R.n)),
        }
    lnn = row_nnz(L.ntok_off)
    lna = row_nnz(L.atok_off)
    rnn = np.diff(np.asarray(R.ntok_off)).astype(np.int32)
    rna = np.diff(np.asarray(R.atok_off)).astype(np.int32)
    r_in2 = rn < R.n2
    r2 = np.where(r_in2, rn, 0)
    r3 = np.where(r_in2, 0, rn - R.n2)
    out = {}
    dot_n, sh_n = _inter(mats["n"], ln, rn)
    dot_a, sh_a = _inter(mats["a"], ln, rn)
    out["ncos"] = dot_n
    out["acos"] = dot_a
    out["nsh"] = sh_n.astype(np.float32)
    out["ash"] = sh_a.astype(np.float32)
    out["njn"], out["nco"], out["ncs"] = jaccard(sh_n, lnn[ln], _rn(rnn, r2, r3))
    out["ajn"], out["aco"], out["acs"] = jaccard(sh_a, lna[ln], _rn(rna, r2, r3))
    lc = np.asarray(L.ctry)[ln]
    rc = np.where(r_in2, np.asarray(R.s2.ctry)[r2], np.asarray(R.s3.ctry)[r3])
    out["csame"] = (lc == rc).astype(np.float32)
    lh = np.asarray(L.hnum)[ln]
    rh = np.where(r_in2, np.asarray(R.s2.hnum)[r2], np.asarray(R.s3.hnum)[r3])
    out["hn_eq"] = ((lh == rh) & (lh > 0)).astype(np.float32)
    out["hn_both"] = ((lh > 0) & (rh > 0)).astype(np.float32)
    lz = np.asarray(L.zpc)[ln]
    rz = np.where(r_in2, np.asarray(R.s2.zpc)[r2], np.asarray(R.s3.zpc)[r3])
    out["zp_eq"] = ((lz == rz) & (lz > 0)).astype(np.float32)
    out["zp_both"] = ((lz > 0) & (rz > 0)).astype(np.float32)
    lf = np.asarray(L.flag)[ln]
    out["f_nonlatin"] = ((lf & 3) != 0).astype(np.float32)
    out["f_noaddr"] = ((lf & 4) != 0).astype(np.float32)
    out["n_len_l"] = lnn[ln].astype(np.float32)
    out["n_len_r"] = _rn(rnn, r2, r3).astype(np.float32)
    out["a_len_l"] = lna[ln].astype(np.float32)
    out["a_len_r"] = _rn(rna, r2, r3).astype(np.float32)
    return out, mats


def _rn(arr, r2, r3):
    return np.where(r2 < len(arr), arr[np.minimum(r2, len(arr) - 1)],
                    arr[np.minimum(r3, len(arr) - 1)])


def _inter(m, ln, rn):
    """One CSR multiply gives both the weighted cosine and the shared-token count."""
    xl, xr = m
    I = xl[ln].multiply(xr[rn])
    dot = np.asarray(I.sum(axis=1)).ravel().astype(np.float32)
    shared = np.diff(I.indptr).astype(np.int32)
    return dot, shared


FEATURE_ORDER = [
    "ncos", "acos", "nsh", "ash", "njn", "nco", "ncs", "ajn", "aco", "acs",
    "csame", "hn_eq", "hn_both", "zp_eq", "zp_both", "f_nonlatin", "f_noaddr",
    "n_len_l", "n_len_r", "a_len_l", "a_len_r",
]


def stack(out):
    return np.column_stack([out[k] for k in FEATURE_ORDER]).astype(np.float32)
