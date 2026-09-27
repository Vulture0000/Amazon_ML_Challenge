# Amazon_ML_Challenge

Business entity resolution over a 3-source marketplace dataset: link
2,206,821 `source1` businesses to `source2` (5,034,616) and `source3`
(5,285,603) records, scored with a macro-averaged F0.5.

Full method, measurements and error analysis: **[`Documentation.md`](Documentation.md)**.

## Result

Honest evaluation on the 220,682 held-out source1 entities, against the **full**
ground truth:

| pipeline | macro F0.5 | precision | recall | singletons kept |
| --- | --- | --- | --- | --- |
| single pass (stage 1 + stage 2) | 0.8532 | 0.9751 | 0.7815 | — |
| \+ targeted recovery pass | 0.9203 | 0.9752 | 0.8427 | 11,191 / 12,268 |
| \+ global conjunctive pass \+ competition ranker | **0.9370** | 0.9784 | 0.8819 | 11,278 / 12,268 |

Retrieval covers 94% of the true pairs, so the remaining error is dominated by
*ranking*: 67% of the loss is entities that predict a partly-correct match.

## Approach

Three candidate pools, each ranked by the model that suits it, merged and
deduplicated:

1. **Base blocking** — name sets, address tokens, house numbers, name prefixes,
   char-4 prefixes, with postings capped. A cheap stage-1 ranker prunes the
   result; its recall is measured on a held-out chunk (**0.9956**), which is what
   caught a silent 7.4% recall collapse.
2. **Targeted recovery** — entities the first pass left empty only, re-blocked on
   a name suffix-4 key and a vowel-stripped address key, with much larger
   posting caps. Strictly additive, so it cannot damage a good prediction.
3. **Global conjunctive pass** — rows sharing *both* recovery keys, keyed exactly
   on `(nsuf_id << 25) | askel_id`. The same coverage as the disjunctive union for
   68.7 candidates per row instead of ~42 billion, which is what makes a global
   pass affordable.

The base pool is ranked by a **competition ranker** (stage 2b): ten features
describing how hard an entity competes for a right record — claim margin and
rank against all other claimants of that record, claimant count, the entity's
runner-up margin, same-source margin, crowd sizes and stage-1 rank. This lifted
held-out average precision 0.98852 → 0.99452. The two recall pools keep the
stage-2 model, because stage 2b only ever trains on base-pool rows and its
`first_pass` column is a shortcut that does not generalise to them.

Thresholds are fitted per pool on the full ground truth
(`15_sweep_v2.py`) and each sits on a broad plateau, so the operating point is
not knife-edge.

## Layout

```
code/business_entity_resolution/   all source; entry point run_all.sh
student_resource/utils/            the challenge's own validator
Documentation.md                   method, measurements, error analysis
output/                            submission TSVs (gitignored; see output/README.md)
work/                              intermediate arrays, 29 GB (gitignored)
```

Numbered scripts are the pipeline stages in order: `01_preprocess` →
`02_blocking` → `04_rank`/`04b_stage1` → `05_fuzzy` → `06_model` →
`07_predict`, then the recall passes `08a_mask`/`08_recover`/`09_conj`, the
competition ranker `stage2b`/`13_score_all`/`14_train2b`, and the measurement
and merge steps `15_sweep_v2`/`16_test_score`/`17_apply_mix`. Files named
`exp_*.py`, `10_rescue.py` and `12_decide.py` are experiments, two of which are
rejected ideas kept for the record.

## Running it

Requires the challenge dataset at `student_resource/dataset/` (not in this
repository) and Python 3.9.

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution
bash run_all.sh train     # fits both models and writes work/sweep_*.txt
bash run_all.sh test      # writes output/*.tsv and validates them
```

`run_all.sh test` reads the two thresholds back out of `work/sweep_*.txt`
instead of hard-coding them, so a refit cannot be merged with a stale operating
point. Preprocessing runs automatically when `work/prep` is missing. Expect
hours and ~34 GB of scratch for a cold run; the intermediate arrays under `work/`
are not versioned, so a fresh clone starts at step 0.
