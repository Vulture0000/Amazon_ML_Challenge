#!/bin/bash
# Full pipeline: blocking -> stage-1 pruning -> fuzzy features -> model ->
# targeted recovery -> global conjunctive pass -> competition ranker -> outputs
#
#   bash run_all.sh [train|test|all]
#
# The train half is what fits the models and picks the operating points; the
# test half only applies them.  The two thresholds that the merge needs are not
# hard-coded: `15_sweep_v2.py` writes them to work/sweep_*.txt and step 6d reads
# them back, so a refit can never be merged with a stale operating point.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
WORK=../../work
SPLIT="${1:-all}"

# 01_preprocess.py builds the country table from the splits it is given and
# overwrites it, so it must run once for all six tables: preprocessing train with
# one country table and test with another would silently change normalisation.
if [ ! -f "$WORK/prep/country_table.txt" ]; then
  echo "=== preprocess: all 6 source tables -> work/prep"
  $PY 01_preprocess.py 2>&1 | tee $WORK/prep.log
else
  echo "=== preprocess: skipped, work/prep already built (delete it to rebuild)"
fi

export CAPS="name_set=2000,addr_set=100,house_number=2000,name_pre=2000,addr_tok=1000,name_tok=2000"
export LEFT_CHUNK=6000
export PAIR_BATCH=3000000

if [ "$SPLIT" = "all" ] || [ "$SPLIT" = "train" ]; then
  echo "=== 0. stage-1 pruning ranker (fit on random train chunks, checked on a held-out chunk)"
  STAGE1_SAMPLE=48000 NEG_PER_CHUNK=120000 S1_ITER=45 $PY 04b_stage1.py

  echo "=== 1. blocking + stage-1 pruning (train)"
  STAGE1_LOAD=1 $PY 04_rank.py train 2>&1 | tee $WORK/rank_train.log

  echo "=== 2. stage-2 string features (train)"
  $PY 05_fuzzy.py train 2>&1 | tee $WORK/fuzzy_train.log

  echo "=== 3. stage-2 model + threshold (train)"
  NEG_SAMPLE=0.35 M2_ITER=400 $PY 06_model.py train 2>&1 | tee $WORK/model_train.log

  echo "=== 3a. extra blocking keys (nsuf / askel)"
  $PY 02b_extra_keys.py --tags train_s1,train_s2,train_s3 2>&1 | tee $WORK/keys.log

  echo "=== 3b. targeted recovery pass on the held-out entities"
  $PY 08a_mask.py --split train --out $WORK/mask_train_valid.npy --valid-only \
      2>&1 | tee $WORK/recv_valid.log
  REC_KEEP=20 $PY 08_recover.py --split train --mask $WORK/mask_train_valid.npy \
      --tag _recv2 --caps 20000,20000 2>&1 | tee -a $WORK/recv_valid.log

  echo "=== 3c. global conjunctive pass over every held-out entity"
  $PY 08a_mask.py --split train --out $WORK/mask_train_allvalid.npy --all-valid
  CONJ_KEEP=20 $PY 09_conj.py --split train --mask $WORK/mask_train_allvalid.npy \
      --tag _conj --cap 50 2>&1 | tee $WORK/conj_train.log

  echo "=== 3d. string features for both recall pools"
  TAG=_recv2 $PY 05_fuzzy.py train 2>&1 | tee $WORK/fuzzy_train_recv.log
  TAG=_conj $PY 05_fuzzy.py train 2>&1 | tee $WORK/fuzzy_train_conj.log

  echo "=== 3e. stage-2 scores for every train candidate (the competition features
  need the whole candidate universe, not just the held-out rows)"
  $PY 13_score_all.py 2>&1 | tee $WORK/score_all.log

  echo "=== 3f. competition ranker (stage 2b) + its scores for every pool"
  NEG_SAMPLE=0.25 M2_ITER=400 $PY 14_train2b.py 2>&1 | tee $WORK/train2b.log

  echo "=== 3g. per-pool operating point on the full ground truth"
  $PY 15_sweep_v2.py --mix base=sv2,rec=sv,conj=sv 2>&1 | tee $WORK/sweep.log
fi

if [ "$SPLIT" = "all" ] || [ "$SPLIT" = "test" ]; then
  echo "=== 4. blocking + stage-1 pruning (test)"
  $PY 04_rank.py test 2>&1 | tee $WORK/rank_test.log

  echo "=== 5. stage-2 string features (test)"
  $PY 05_fuzzy.py test 2>&1 | tee $WORK/fuzzy_test.log

  echo "=== 6. base submission from stage 2 (also the target set for 6b)"
  $PY 07_predict.py test 2>&1 | tee $WORK/predict_test.log

  echo "=== 6a. extra blocking keys (nsuf / askel)"
  $PY 02b_extra_keys.py --tags test_s1,test_s2,test_s3 2>&1 | tee $WORK/keys_test.log

  echo "=== 6b. targeted recovery pass (entities the first pass left empty)"
  $PY 08a_mask.py --split test --out $WORK/mask_test.npy 2>&1 | tee $WORK/recv_test.log
  LEFT_CHUNK=5000 REC_KEEP=20 $PY 08_recover.py --split test \
      --mask $WORK/mask_test.npy --tag _recv --caps 20000,20000 \
      2>&1 | tee -a $WORK/recv_test.log
  TAG=_recv $PY 05_fuzzy.py test 2>&1 | tee $WORK/fuzzy_test_recv.log

  echo "=== 6c. global conjunctive pass over every test entity"
  $PY 08a_mask.py --split test --out $WORK/mask_test_all.npy --all-rows
  CONJ_KEEP=20 LEFT_CHUNK=20000 $PY 09_conj.py --split test \
      --mask $WORK/mask_test_all.npy --tag _conj --cap 50 \
      2>&1 | tee $WORK/conj_test.log
  TAG=_conj $PY 05_fuzzy.py test 2>&1 | tee $WORK/fuzzy_test_conj.log

  echo "=== 6d. stage-2 scores for the recall pools, stage-2b features over the
  union of all three pools, stage-2b scores for the base pool"
  $PY 16_test_score.py 2>&1 | tee $WORK/test_score.log

  echo "=== 6e. merge (thresholds from work/sweep_base-sv2+rec-sv+conj-sv_*.txt)"
  SWEEP=$WORK/sweep_base-sv2+rec-sv+conj-sv_base-rec-conj.txt
  BASE_THR=$(sed -n 's/^base=//p' "$SWEEP")
  REC_THR=$(sed -n 's/^rec=//p' "$SWEEP")
  CONJ_THR=$(sed -n 's/^conj=//p' "$SWEEP")
  $PY 17_apply_mix.py --split test \
      --spec "base:_sv2:$BASE_THR,_recv:_sv1:$REC_THR,_conj:_sv1:$CONJ_THR" \
      2>&1 | tee $WORK/apply_test.log

  echo "=== 7. validate"
  $PY ../../student_resource/utils/validate_submission.py \
      --matching ../../output/matching_results.tsv \
      --candidate ../../output/candidate_pairs.tsv \
      --test-dir ../../student_resource/dataset/test --check-ids
fi
