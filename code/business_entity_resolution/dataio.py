"""
dataio.py -- access layer for the preprocessed arrays written by 01_preprocess.py
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP = os.path.join(ROOT, "work", "prep")
DATA_DIR = os.path.join(ROOT, "student_resource", "dataset")
WORK = os.path.join(ROOT, "work")

SPLITS = {
    "train": ["train_s1", "train_s2", "train_s3"],
    "test": ["test_s1", "test_s2", "test_s3"],
}


def load_country_table():
    m = {}
    with open(os.path.join(PREP, "country_table.txt")) as f:
        for line in f:
            code, name = line.rstrip("\n").split("\t")
            m[int(code)] = name
    return m


class Source:
    """One preprocessed source file (train_s2, test_s1, ...)."""

    def __init__(self, tag, mmap=True):
        self.tag = tag
        m = "r" if mmap else None
        self.ids = np.load(os.path.join(PREP, f"{tag}_id.npy"), mmap_mode=m)
        self.ctry = np.load(os.path.join(PREP, f"{tag}_ctry.npy"), mmap_mode=m)
        self.flag = np.load(os.path.join(PREP, f"{tag}_flag.npy"), mmap_mode=m)
        self.ntok_off = np.load(os.path.join(PREP, f"{tag}_ntok.off.npy"), mmap_mode=m)
        self.ntok = np.load(os.path.join(PREP, f"{tag}_ntok.npy"), mmap_mode=m)
        self.atok_off = np.load(os.path.join(PREP, f"{tag}_atok.off.npy"), mmap_mode=m)
        self.atok = np.load(os.path.join(PREP, f"{tag}_atok.npy"), mmap_mode=m)
        self.npre_off = np.load(os.path.join(PREP, f"{tag}_npre.off.npy"), mmap_mode=m)
        self.npre = np.load(os.path.join(PREP, f"{tag}_npre.npy"), mmap_mode=m)
        self.hnum = np.load(os.path.join(PREP, f"{tag}_hnum.npy"), mmap_mode=m)
        self.zpc = np.load(os.path.join(PREP, f"{tag}_zpc.npy"), mmap_mode=m)
        self.name_off = np.load(os.path.join(PREP, f"{tag}_name.off.npy"), mmap_mode=m)
        self.addr_off = np.load(os.path.join(PREP, f"{tag}_addr.off.npy"), mmap_mode=m)
        self._blob = open(os.path.join(PREP, f"{tag}_name.blob"), "rb")
        self._ablob = open(os.path.join(PREP, f"{tag}_addr.blob"), "rb")
        self.n = len(self.ids)

    def __len__(self):
        return self.n

    def name(self, i):
        """Canonical name string of record i."""
        o = self.name_off
        self._blob.seek(int(o[i]))
        return self._blob.read(int(o[i + 1]) - int(o[i])).decode("utf-8", "replace")

    def names(self, idx):
        """Vectorized-ish: list of canonical names for an array of row indices."""
        return self._read(self._blob, self.name_off, idx)

    def addr(self, i):
        return self._read_one(self._ablob, self.addr_off, i)

    def addrs(self, idx):
        return self._read(self._ablob, self.addr_off, idx)

    @staticmethod
    def _read_one(blob, off, i):
        blob.seek(int(off[i]))
        return blob.read(int(off[i + 1]) - int(off[i])).decode("utf-8", "replace")

    @classmethod
    def _read(cls, blob, off, idx):
        return [cls._read_one(blob, off, i) for i in idx]

    def row_tokens(self, field, i):
        if field == "n":
            off, tok = self.ntok_off, self.ntok
        else:
            off, tok = self.atok_off, self.atok
        return tok[off[i]:off[i + 1]]

    def ntok_csr(self):
        return self.ntok_off, self.ntok

    def atok_csr(self):
        return self.atok_off, self.atok


def source_id(tag):
    return int(tag[-1])


def eid_str(source, num):
    return f"S{source}-{int(num)}"
