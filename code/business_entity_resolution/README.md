# Business Entity Resolution - Day 1 Baseline

## Setup

```bash
pip install -r requirements.txt
```

## Data Layout

Place the competition data in the project root:
```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

## Running

### EDA
```bash
python3 code/business_entity_resolution/src/eda.py dataset/train
```

### Full Day 1 Pipeline (train + test)
```bash
python3 code/business_entity_resolution/src/run_day1.py --mode both --top-k 30
```

### Validate submission
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline

1. **Normalize** - lowercase, strip accents, extract legal suffixes, expand abbreviations, extract postal/house numbers
2. **Block** - TF-IDF char n-gram on names, TF-IDF word on name+address, shared rare tokens, postal+name
3. **Score** - rapidfuzz similarities (Jaro-Winkler, token_set, partial_ratio, ratio, token_sort) on name and address
4. **Threshold** - grid-search t_match and t_empty on training F0.5
5. **Conflict resolution** - assign each S2/S3 candidate to its highest-scoring S1 only
