# Business Entity Resolution

Matches every Source 1 business to its records in Sources 2 and 3 and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Setup

Python 3.11 or later.

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Data layout

Run from the submission root with the competition data in `dataset/`:

```
dataset/
  train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## Reproduce end to end

```bash
python -u code/business_entity_resolution/src/run_pipeline.py --mode both
```

`--mode train` trains on the training data and writes the models to `output/`;
`--mode test` loads them and writes the two submission files. `--mode both` does both.

Defaults: `--k 10 --kc 5 --df-cap 2000 --filter-recall 0.995`. Test mode must use
the same blocking settings as training (checked at start-up).

On a 4-core, 30GB Kaggle CPU notebook, the full run takes about 3 to 3.5 hours.
Every stage is cached under `cache/`, so a rerun resumes from the last finished
stage; delete `cache/` to start clean.

Validate the output:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline

| Step | Module | What it does |
|---|---|---|
| 1. Normalize | `normalize_fast.py` | Lower-case, strip accents, canonicalize legal forms, abbreviations, directionals and number formats; extract postal code and house number |
| 2. Block | `blocking_fast.py` | IDF-weighted token overlap per country (sparse matrix product): top-k Source 2/3 records per Source 1 entity, plus top-kc Source 1 entities per Source 2/3 record |
| 3. Filter candidates | `features.py`, `ml_scorer.py` | Small LightGBM on 18 cheap features scores every blocked pair; pairs below a threshold chosen to keep 99.5% of blocking's true pairs are dropped |
| 4. Features | `features.py` | 48 pair features for the filtered pairs, plus the filter score and competition features computed over the full blocked set |
| 5. Match | `ml_scorer.py` | LightGBM matcher; threshold and assignment rule tuned for macro F0.5 on held-out Source 1 entities; each Source 2/3 record is assigned to at most one Source 1 entity |

`candidate_pairs.tsv` is the filtered set from step 3: exactly the pairs the
matcher scores. `matching_results.tsv` is a subset of it.

## Outputs

| File | Contents |
|---|---|
| `output/matching_results.tsv` | Final matches |
| `output/candidate_pairs.tsv` | Candidate set fed to the matcher |
| `output/filter_model.txt`, `output/filter_params.json` | Candidate filter model and threshold |
| `output/model.txt`, `output/selection_params.json` | Matcher model and tuned selection rule |
