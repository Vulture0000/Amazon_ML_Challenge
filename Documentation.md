# ML Challenge 2026: Business Entity Resolution — Solution

**Team Name:** vulture
**Team Members:** —
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A three-stage cascade: six-family inverted-index blocking generates a candidate
set, a gradient-boosted *pruning* ranker cuts each entity's candidates to its top
20, and a second gradient-boosted model scores the survivors on cheap sparse
TF-IDF features plus rapidfuzz string-similarity features. The final decision
threshold is chosen to maximise the competition's macro per-entity F0.5
directly, which also handles the singleton rule (an entity with no match must
stay unmatched, otherwise it scores 0).

A fourth, *targeted* stage was added at the end: entities that the cascade
leaves with no match at all are re-blocked with two extra hashed keys (a name
suffix-4 and an address-skeleton token) and re-scored with the same models.
Because those entities have no existing prediction, the merge can only add
matches, never change one. On held-out entities scored against the full ground
truth this lifts macro F0.5 from 0.8532 to **0.9203**, and on test it turns
113,543 empty rows into matched ones.

---

## 2. Methodology

### 2.1 Problem Analysis

* **Metric shape.** Scoring is macro (per Source-1 entity) F0.5, and a Source-1
  entity with *no* true match is scored 1.0 if we predict nothing and 0.0 if we
  predict anything. Precision is weighted 4x more than recall, so a wrong merge
  on a true singleton costs a full point while a missed match on a 1-match
  entity costs far less. The threshold must therefore be tuned on the metric,
  not at 0.5 probability.
* **Name noise.** Legal-form and suffix variation (`Pvt Ltd` / `Private Limited`
  Company / `P.L.C.`), DBA/domain names (`lordenterprises com` for "Lord
  Enterprises"), CJK transliterations, non-Latin scripts, and token
  re-orderings ("Smith John A" / "John A Smith").
* **Address noise.** Abbreviation expansion (`St`/`Street`, `Ave`/`Avenue`),
  missing house numbers, unit/suite noise, and re-ordering of the city/state
  fields between sources.
* **Country.** Train is USA-only; the test split adds **France** (and other
  countries). Only 0.13% of positive training pairs are cross-country, so
  country is a feature, never a hard filter.
* **Class imbalance.** The raw blocking output contains ~0.25% positives, and
  123,247 of 2,206,821 train entities (5.6%) are singletons.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted cascade (hybrid)
**Core Innovation:** a *measured* blocking design — every key family is scored
for true-pair coverage on a 103,419-pair sample before being kept, and a
dedicated pruning ranker guarantees that the cheap-to-score stage does not lose
recall relative to the blocking stage.

---

## 3. Candidate Generation (Blocking)

Six key families, each an inverted index over the concatenated Source-2/3
records, each key bucket capped to control candidate volume:

| family | key | cap | true-pair agreement (103k sample) |
| --- | --- | --- | --- |
| `name_set` | sorted name-token set | 2000 | 0.5231 |
| `addr_set` | sorted address-token set | 100 | 0.6695 (hnum) |
| `house_number` | house-number token | 2000 | 0.6695 |
| `name_pre` | name prefix-4 (char) | 2000 | 0.9231 |
| `addr_tok` | address token | 1000 | 0.9563 |
| `name_tok` | name token | 2000 | 0.8608 |

* **Candidate pairs generated:** ~2.56e9 raw (≈1160 per Source-1 entity) on the
  train split; the caps keep a 2000-entity-per-key explosion (e.g. the token
  `main` or `st`) from dominating the set.
* **How true matches were not lost:**
  1. *Union coverage was measured, not assumed.* The sample experiment showed
     `{name_pre, address-skeleton, house_number, name_set}` reaches 1.0000
     union coverage on the sampled true pairs, so the families are complementary
     and none of them is redundant.
  2. *Cheap family features keep the key origin.* Each candidate carries a 6-bit
     family mask, so the model can learn which key produced it; dropping a key
     would silently remove candidates the model relies on.
  3. *No hard filters.* Country, ZIP and length filters are never used as
     blocking filters (only as features) because 0.13% of positive pairs are
     cross-country and non-matching lengths are common.

---

## 4. Matching Model

**Features (39 total, all cheap and vectorised):**

* *Name (10):* TF-IDF cosine, shared-token count, Jaccard, containment,
  country-of-name match, flag bits (non-Latin, no address), name length.
* *Address (11):* TF-IDF cosine, shared-token count, Jaccard, containment,
  house-number equality, ZIP equality, address length, first-character match.
* *Pair-shape (6):* one indicator per blocking family.
* *String similarity (12, rapidfuzz, element-wise `cpdist`):* Jaro-Winkler,
  WRatio, `token_set_ratio`, `token_sort_ratio` on both name and address, plus
  length ratios, digit-sequence equality and address first-character equality.

**Model type:** `sklearn.ensemble.HistGradientBoostingClassifier` (LightGBM is
unavailable in this environment — its wheel needs `@rpath/libomp.dylib`).
Stage 1 (pruning): 45 iterations, 63 leaves, fitted on 1.1M rows containing
*all* positives from 8 random chunks plus 120k negatives per chunk. Stage 2
(final): 600 iterations, 63 leaves, 10% of Source-1 entities held out.

**Threshold selection method:** direct maximisation of macro F0.5 over a
190-point score-quantile grid on the held-out entity split (never 0.5). The grid
is evaluated against the **full ground truth** restricted to the held-out
entities, not against the candidate labels: deriving the truth from the
candidates credits every entity whose true pairs were never proposed as a
perfect singleton, which inflated an earlier report of this pipeline to
0.95393 when the honest value is 0.85320. The operating point is
`thr = 0.669231` (0.653846 is the sweep optimum; the two differ by 0.00005).

---

## 5. Results & Error Analysis

* **Pruning ranker, held-out chunk:** ROC-AUC 0.9996, top-20 positive recall
  **0.9956** — i.e. the cheap stage-1 features lose only 0.44% of the recall
  that blocking found.
* **Stage 2, honest held-out macro F0.5 (full ground truth, 220,682 held-out
  entities): 0.8532** (micro 0.9291, P 0.9751, R 0.7815, 612,163 predicted
  pairs). 115,696 true pairs are missed: 61,026 of them belong to entities that
  do predict (wrongly or only partially) and 54,670 to entities that predict
  nothing at all.
* **Blocking ceiling:** the raw candidate set contains 84.9% of all true pairs
  (6,483,513 of 7,638,365; 6,494,936 with the un-pruned first family set). This
  is the dominant error source: the remaining 15% are pairs with no shared name
  token, no shared address token, no shared prefix-4 and no house number
  (re-typed or truncated addresses).

### 5.1 Second-chance recovery pass (`08_recover.py`)

Because 47% of the missed pairs belong to entities that score a flat 0.0, a
*targeted* extra blocking pass is nearly free in terms of risk: it can only add
a prediction to an entity that currently has none.

* **New families** (`02b_extra_keys.py`, hashed, no vocabulary):
  `nsuf` = last four characters of every canonical name token;
  `askel` = address token with vowels removed, digits spelled out, truncated to
  six characters. On 30,000 sampled missed pairs the union of the two keys
  covers **98.7%** of them (individually 81.1% and 85.9%).
* **Target set:** only entities with no pair above the stage-2 threshold —
  31,718 held-out train entities, 225,053 test entities.
* **Cost control:** posting-list caps of 20,000 and stage-1 top-20 pruning,
  which is what makes the pass affordable: raising the caps from 2,000 to 20,000
  lifts the recovery of missed pairs from 21% to 87.7% and the honest macro
  from 0.8678 to 0.9203, at 30x the raw candidates (0.65 GB/s of feature
  building). The raw volume is bounded by the *small* target set, not by the
  1.7M test entities, which is why a global re-blocking (≈42 billion raw
  candidates) was rejected.
* **Result (honest, full ground truth, held-out entities): macro F0.5
  0.8532 → 0.9203 (+0.067)** at a recovery threshold of 0.7946 (the curve is
  flat between 0.70 and 0.88, so the operating point is not knife-edge);
  micro P 0.9752, R 0.8427, 660,002 predicted pairs, 11,191/12,268 true
  singletons preserved. Raising the pruning depth from 20 to 40 changes nothing
  (+0.0001), so 20 was kept.
* **Not done, deliberately:** a second pass over already-predicted entities can
  address at most 8,862 of the remaining missed pairs (7.7%), and those live in
  only 3,439 entities — while every other low-confidence entity that already
  predicts correctly can only *lose* from extra candidates. The risk/reward does
  not justify the extra hours of compute.
* **Deployed on test:** 225,053 target entities → 4,690,186,235 raw candidates
  → 4,404,874 after stage-1 pruning (19.57 per target, 94 min), of which
  350,525 across 113,543 entities clear the recovery threshold. The merge is
  strictly additive: 1,507,491 → 1,621,034 entities with a match, 5,023,205 →
  5,373,730 pairs, 0 entities lost or modified. `candidate_pairs.tsv` now holds
  the union of both candidate sets (35,358,383 pairs, 963,043 duplicates
  dropped) so that every final match is backed by a published candidate.
  `validate_submission.py --check-ids` passes with no warnings.
  `output/baseline_matching_results.tsv` and `output/baseline_candidate_pairs.tsv`
  keep the pre-recovery submission for rollback.
* **Common false positives:** near-duplicate *chain* entries (same name, same
  street, different suite/PO box) where the only distinguishing token is a unit
  number; and French transliteration pairs where two different streets collapse
  to the same normalised token set.
* **Common false negatives:** domain/DBA names with no lexical overlap with the
  registered name; addresses truncated to a PO box.

### 5.2 Conjunctive pass and the competition ranker (deployed)

With retrieval covering 94% of the true pairs, the remaining loss is a *ranking*
problem, and the loss decomposition says so precisely (full ground truth, 220,682
held-out entities, macro 0.9265):

| bucket | entities | macro lost |
|---|---|---|
| matched but **partially** (mean f 0.861) | 78,415 | 0.0494 (67%) |
| matched but nothing predicted | 4,231 | 0.0192 (26%) |
| true singletons wrongly filled | 1,091 | 0.0049 (7%) |

Two things follow, and both were built.

**A global conjunctive pass** (`09_conj.py`). The disjunctive recovery pass needs
≈42 billion raw candidates to run globally, but its coverage comes from cases
sharing *one* of the two keys, and requiring *both* is enormously more selective:
rows share a cross key `(nsuf_id << 25) | askel_id` iff they share a name
suffix-4 *and* an address skeleton token, covering 66.9% of missed pairs for
68.7 raw candidates per row. The key must stay exact — hashing it to 25 bits to
match the other families collides badly and inflates the candidate count ~100x
(`exp_conj.py`/`exp_conj2.py`). A cap-50 validation pass over all held-out
entities produced 4,123,163 candidates with 625,311 positives, and adds
**+0.0061** on top of the recovery pass (0.9309 → 0.9370 with the ranker below).

**A competition ranker** (`stage2b.py`, `14_train2b.py`, trained on the same rows
and budget as stage 2 with ten extra columns). No pairwise string comparison can
express *how hard an entity is competing for a right record*, so stage 2b adds
claim margin and claim rank against every other entity that wants the same right
record, the number of claimants, the entity's own runner-up margin, its
same-source margin, crowd sizes, and the stage-1 rank. Measured at equal
training budget: held-out average precision 0.98852 → **0.99452**, and on exactly
the bucket that holds two thirds of the loss (entities with a true match in the
pool) macro 0.92803 → **0.94655**.

**The pools are ranked by different models, on purpose.** Stage 2b has only ever
been trained on base-pool rows, so on the recall pools — by construction full of
candidates the first pass rejected — it collapses, and their thresholds are
pushed to 0.99 (`15_sweep_v2.py` measured 0.8746 for a naive stage-2b
everywhere). The deployed merge therefore scores the base pool with stage 2b and
both recall pools with stage 2:

| pool | model | threshold | held-out macro |
|---|---|---|---|
| base blocking | stage 2b | 0.810 | 0.8640 (stage 2: 0.8532) |
| + recovery (`_recv`) | stage 2 | 0.785 | 0.9309 |
| + conjunctive (`_conj`) | stage 2 | 0.770 | **0.93697** |

versus **0.9265** for stage 2 everywhere, and 0.8532 for the original single
pass. Micro precision 0.9784, recall 0.8819. Each threshold sits on a broad
plateau (base 0.70–0.90 spans 0.9341–0.9370, recall 0.70–0.90 spans
0.9356–0.9370), so the operating point is not a knife-edge.

A per-entity decision layer (accept rank 1 above `t1`, ranks 2+ above `t2` when
the best is above `t3`), cross-fitted on the held-out split, was also tried and
**rejected**: it collapses to a single global threshold and gains +0.0005, i.e.
the score profile carries no information about the number of true matches beyond
the score itself. A rescue-threshold sweep for the empty entities was likewise
worthless (+0.0004): 20,366 of the 31,718 empty held-out entities do have a true
match, but only 1,893 have a true *best* candidate.

**Deployed on test** (`16_test_score.py`, `17_apply_mix.py`): 32,133,890
conjunctive candidates, stage-2b features over the 68.4M-row union of all three
pools, then the merge deduplicating *before* thresholding so a pair cannot rescue
itself from a second pool. 5,619,934 pairs over 1,686,891 entities (45,653 empty),
up from 5,373,730 over 1,621,034. `validate_submission.py --check-ids` passes with
no warnings. The pre-recovery rollback files are unchanged; the 0.9203 recovery
submission is reproducible with
`08c_apply.py --split test --tag _recv --rec-thr 0.794601`.

---


The design separates *recall-critical* work (blocking, then a pruning ranker
whose recall is measured on held-out data) from *precision-critical* work (the
final model plus a metric-aware threshold). Measuring the pruning recall on a
held-out chunk rather than trusting the model turned a silent 7.4% recall
collapse into 99.6%, which was the single largest fix in the project.

---

## Appendix

### A. Code Artefacts

All source lives in `code/business_entity_resolution/`; entry point
`run_all.sh [train|test]`, which runs the stages below in order.

| file | role |
| --- | --- |
| `erlib.py` | Unicode/name/address normalisation (NFKD, transliteration, legal suffixes, address abbreviations, French patterns) |
| `01_preprocess.py` | parallel preprocessing → `work/prep/*.npy` CSR arrays + string blobs |
| `dataio.py`, `common.py` | array/blob access, concatenated S2+S3 `Right` view, ground truth, macro-F0.5 metric |
| `02_blocking.py` | reusable inverted-index primitives (`build_index`, `key_counts`, `expand`, `filter_csr`, `set_key`, `num_key`) |
| `features.py` | 21 vectorised sparse TF-IDF / overlap features |
| `04_rank.py` | chunked blocking + stage-1 pruning → `work/pairs_{split}_*.npy` |
| `04b_stage1.py` | fits + saves the pruning ranker, and reports its **held-out pruning recall** |
| `05_fuzzy.py` | 12 rapidfuzz string-similarity features |
| `06_model.py` | stage-2 model, metric-aware threshold sweep, macro-F0.5 report |
| `07_predict.py` | inference + writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` |
| `02b_extra_keys.py` | builds the `nsuf` / `askel` hashed CSR keys used by the recovery pass |
| `08a_mask.py` | selects the entities the first pass left empty (the recovery target set) |
| `08_recover.py` | targeted second-chance blocking + stage-1 pruning on that target set |
| `08b_recover_score.py` | scores the recovery candidates and sweeps the merged, honest macro F0.5 |
| `08c_apply.py` | merges both passes and writes the final TSVs |
| `09_conj.py`, `09b_conj_score.py` | global *conjunctive* blocking pass (exact `nsuf`×`askel` cross keys) and its honest merged scoring |
| `stage2b.py` | the ten competition/crowd features, shared by training and inference, plus the per-pool feature and score caches |
| `11_feat2.py` | measured comparison of feature sets at equal training budget (AP + macro on the loss-bearing bucket) |
| `13_score_all.py` | one stage-2 pass over all 40.2M train candidates (the competition features need the whole universe) |
| `14_train2b.py` | fits the stage-2b ranker and caches its scores for every pool |
| `15_sweep_v2.py` | per-pool threshold fit on the full ground truth, any mix of models per pool |
| `16_test_score.py` | test-side stage-2b features and scores |
| `17_apply_mix.py` | merges any number of pools, one model and threshold each, deduplicating before thresholding |
| `10_rescue.py`, `12_decide.py` | rejected ideas kept for the record: rescue-threshold sweep, cross-fitted per-entity decision layer |
| `exp_miss2.py`, `exp_miss3.py`, `exp_miss4.py` | missed-pair analysis: key coverage, full-truth metric, confidence buckets |
| `exp_block.py`, `exp_keys.py`, `exp_union.py`, `diag_misses.py`, `test_feats.py` | the measurement scripts behind every design decision above |

Environment: Python 3.9, numpy/scipy/scikit-learn/rapidfuzz/pandas/pyarrow
(`requirements.txt`).
