"""Quick data profiling: cardinality, label stats, noise patterns."""
import sys
from collections import Counter

import pandas as pd

D = "student_resource/dataset"

print("=== row counts / cardinality ===")
for split in ("train", "test"):
    for s in (1, 2, 3):
        p = f"{D}/{split}/{'train' if split=='train' else 'test'}_source{s}.tsv"
        df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
        n = len(df)
        print(f"{split} source{s}: rows={n} uniq_name={df.business_name.nunique()} "
              f"uniq_addr={df.business_address.nunique()} countries={df.country.value_counts().to_dict()}")
        print("   empty name:", int((df.business_name.str.strip() == "").sum()),
              " empty addr:", int((df.business_address.str.strip() == "").sum()),
              " name len med:", int(df.business_name.str.len().median()))
        del df

print("\n=== ground truth ===")
gt = pd.read_csv(f"{D}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
print("rows:", len(gt), "uniq s1:", gt.source1_entity_id.nunique())
n = gt.matched_entity_ids.str.count(",").fillna(0) + (gt.matched_entity_ids.str.strip() != "").astype(int)
print("matches per S1 -> mean %.3f" % n.mean())
print(n.value_counts().head(12).to_dict())
# how many predicted match S2 vs S3
has2 = gt.matched_entity_ids.str.contains("S2-").sum()
has3 = gt.matched_entity_ids.str.contains("S3-").sum()
print("S1 with >=1 S2 match:", int(has2), " with >=1 S3 match:", int(has3))
print("singletons (no match):", int((gt.matched_entity_ids.str.strip() == "").sum()))
print("total positive pairs:", int(n.sum()))

print("\n=== sample positives ===")
s1 = pd.read_csv(f"{D}/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
s2 = pd.read_csv(f"{D}/train/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
s3 = pd.read_csv(f"{D}/train/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
allrec = pd.concat([s1, s2, s3])

import random
random.seed(0)
pos = gt[gt.matched_entity_ids.str.strip() != ""]
for _, row in pos.sample(8, random_state=0).iterrows():
    a = allrec.loc[row.source1_entity_id]
    print("-" * 100)
    print("S1", a.business_name, "|", a.business_address, "|", a.country)
    for mid in row.matched_entity_ids.split(","):
        b = allrec.loc[mid]
        print("   ", mid[:3], b.business_name, "|", b.business_address, "|", b.country)
