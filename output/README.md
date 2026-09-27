# Output

The two TSVs here are the submission. They are build outputs of
`code/business_entity_resolution/run_all.sh test` and are **not** versioned: at
762 MB and 433 MB they are far past GitHub's 100 MB per-file limit. The 20-line
`sample_*.tsv` files in this directory show the exact format.

| file | rows | content |
| --- | --- | --- |
| `matching_results.tsv` | 1,732,544 | `source1_entity_id`, `matched_entity_ids` (comma-separated, right-record ids) |
| `candidate_pairs.tsv` | 1,732,544 | every candidate pair considered, so each match is backed by a published candidate |
| `sample_matching_results.tsv` | 60 | head of the real file |
| `sample_candidate_pairs.tsv` | 20 | head of the real file |

Current deployed run:

- 5,619,934 matched pairs over 1,686,891 entities; 45,653 entities left empty.
- 57,443,906 unique candidate pairs.
- `validate_submission.py --check-ids` passes with no warnings.

`baseline_matching_results.tsv` / `baseline_candidate_pairs.tsv` are the
pre-recovery rollback (single-pass model, honest macro 0.8532).
