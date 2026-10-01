# Business Entity Resolution

Matches every Source 1 business to its records in Sources 2 and 3 and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Setup

Python 3.11 or later.

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Data layout

The competition data is not in the submission zip. Unzip the submission, add the
data as `dataset/` next to `output/` and `code/`, and run every command from that
top-level folder (the submission root):

```
<team_name>_submission/                 <- run commands from here
├── output/                             <- matching_results.tsv, candidate_pairs.tsv
├── code/business_entity_resolution/    <- src/, README.md, requirements.txt
├── Documentation_template.md
└── dataset/                            <- add the competition data here
    ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
    └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## Reproduce end to end

```bash
python -u code/business_entity_resolution/src/run_pipeline.py --mode both
```

This regenerates both files in `output/` from the training and test data, using
only the code in `code/business_entity_resolution/`. `--mode train` trains on the
training data and writes the models to `output/`; `--mode test` loads them and
writes the two submission files. `--mode both` does both.

Defaults: `--k 10 --kc 5 --df-cap 2000 --filter-recall 0.995`. Test mode must use
the same blocking settings as training (checked at start-up).

On a 4-core, 30GB Kaggle CPU notebook, the full run took 207 minutes.
Every stage is cached under `cache/` at the submission root, so a rerun resumes
from the last finished stage; delete `cache/` to start clean.

Validate the output with the organizers' validator, `utils/validate_submission.py`
from the challenge's `student_resource/` folder (it is not part of this package):

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

The submission zip's `output/` holds the two submission files. A run also writes
the trained models and tuned parameters there.

| File | Contents |
|---|---|
| `output/matching_results.tsv` | Final matches |
| `output/candidate_pairs.tsv` | Candidate set fed to the matcher |
| `output/filter_model.txt`, `output/filter_params.json` | Candidate filter model and threshold |
| `output/model.txt`, `output/selection_params.json` | Matcher model and tuned selection rule |
