# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TODO  
**Team Members:** TODO  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A blocking + classifier pipeline: normalization of name and address variants,
two-way IDF token blocking, a learned candidate filter that cuts the candidate set
to 5.3 Source 2/3 records per Source 1 entity, and a LightGBM matcher whose
threshold and one-owner assignment rule are tuned directly for macro F0.5. The key
ideas are blocking from the Source 2/3 side as well (every record has at most one
owner), a second, learned filtering stage whose output is exactly what the
matcher scores, and normalization rules derived from the French noise patterns
the training data does not contain. Final: validation macro F0.5 0.9511,
public leaderboard 0.932.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** training has 2.21M Source 1 entities and 10.32M Source 2+3 records
  with 7.64M true pairs (3.46 per Source 1 entity); 5.58% of Source 1 entities
  have no match. Test has 1.73M Source 1 entities and 9.97M Source 2+3 records.
- **One owner per record:** Source 1 is deduplicated, so every Source 2/3 record
  belongs to at most one Source 1 entity.
- **Crowding:** chains, generic names ("Club", "Amicale", "Ecole", "Traders") and
  many businesses on the same street make a Source 1 entity's own nearest
  neighbours unreliable: even the 20 best-scoring Source 2/3 records per entity
  contain only 90% of the true pairs.
- **Name noise:** legal forms in different spellings and positions (Pvt Ltd /
  Private Limited, S.A.R.L. / SARL, "SARL Dupont", "AS SARL Sportive"),
  & / and / et, transliterations (Shri / Sri / Shree), "trading as" names, word
  order, typos, inserted country words ("Petite Services (France)").
- **Address noise:** abbreviations (St / Street, R. / Rue, Crs / Cours, Opp / Opposite),
  components reordered ("Hauts-de-France, 37 R Chanzy, Lille"), region or state
  included, omitted or swapped ("Hauts-de-France" / "Nord"), number formats
  ("No. 146", "Nº 51", "0021", "12bis"), landmarks ("Near SBI ATM").
- **France:** absent from training (259K Source 1 and 1.43M Source 2+3 test
  records). French addresses have no postal code, so the house number is the main
  way to tell apart businesses on one street.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier, with a learned second blocking stage  
**Core Innovation:** Two-way blocking that uses the one-owner constraint, plus a
learned candidate filter whose output is the exact candidate set the matcher scores
(5.3 per Source 1 entity), with the matcher's competition features computed over
the full blocked set so filtering does not hide rival candidates.

Pipeline (`code/business_entity_resolution/src/`):

| Step | Module | Output |
|---|---|---|
| 1. Normalize names and addresses | `normalize_fast.py` | Canonical name, legal form, acronym, address, postal code, house number |
| 2. Two-way token blocking | `blocking_fast.py` | 30.8 pairs per Source 1 entity (test) |
| 3. Candidate filter | `features.py`, `ml_scorer.py` | 5.3 pairs per Source 1 entity → `candidate_pairs.tsv` |
| 4. Pair features | `features.py` | 57 features per candidate |
| 5. Matcher and selection | `ml_scorer.py` | `matching_results.tsv` |

Normalization rules apply identically to every record regardless of country; the
country label only partitions blocking, as an open set of labels. Only the
provided data is used: no external data, APIs or pretrained models.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** separate vocabularies for name words, address words,
  postal code, name word bigrams, address word bigrams, name character trigrams,
  consonant-skeleton phonetic keys, and name-word@postal composites. Tokens in more
  than 2,000 Source 2/3 records of a country are dropped; the rest are weighted by
  IDF, so a pair's blocking score is the IDF mass of the tokens it shares.
  Scores come from a chunked sparse matrix product per country on 4 processes; no
  all-pairs comparison is ever made.
- **Two directions:** the 10 best Source 2/3 records for each Source 1 entity,
  united with the 5 best Source 1 entities for each Source 2/3 record.
- **Learned candidate filter:** a LightGBM model (63 leaves) scores every blocked
  pair on 18 cheap, scale-free features (relative blocking score and rank from both
  directions, candidate counts, name token-set and Jaro-Winkler similarity,
  address token-set similarity, postal and house-number equality, source, and each
  pair's rank and gap among its Source 1 entity's and its Source 2/3 record's
  candidates). The raw blocking score is left out because its scale depends on how
  many records a country has, which differs for France.
- **Candidate pairs generated:**

  | | Training | Test |
  |---|---|---|
  | After two-way blocking | 57,105,758 (25.9 per Source 1 entity) | 53,413,632 (30.8 per Source 1 entity) |
  | After candidate filter = `candidate_pairs.tsv` | 9,755,277 (4.4 per Source 1 entity) | **9,211,625 (5.3 per Source 1 entity)** |

  For scale, training data has 3.46 true matches per Source 1 entity.
- **How you ensured true matches were not lost:**
  - The reverse direction recovers matches crowded out of an entity's own
    top-k: reverse top-1 alone reaches 0.9104 recall, more than forward top-10
    (0.8898); reverse top-5 reaches 0.9473 and the union 0.9490.
  - IDF weighting plus composite and phonetic tokens keep rare, discriminative
    evidence while generic words are dropped; the normalization rules make
    variants of one business share tokens.
  - The filter threshold is set on held-out Source 1 entities to keep 99.5% of
    the true pairs blocking found; recall goes from 0.9490 to 0.9446.
  - Recall was measured at every stage against the ground truth on each run.

| Blocking configuration | Recall (training) |
|---|---|
| Forward top-20 only (baseline) | 0.9034 |
| Forward top-10 + reverse top-3 (v3) | 0.9409 |
| Forward top-10 + reverse top-5 (final) | 0.9490 |
| Final, after candidate filter | 0.9446 |

---

## 4. Matching Model

**Features used (57):**
- Name features: ratio, token-set, token-sort, partial ratio and Jaro-Winkler on
  the core name; token-set on the full name; TF-IDF cosine on name words and name
  character 3-grams; exact match, first/last token match, lengths, token counts,
  acronym matches, legal-form match.
- Address features: ratio, token-set, partial ratio and Jaro-Winkler; TF-IDF
  cosine on address words; postal-code and house-number equality and presence;
  shared count, Jaccard and joint presence of every number in the address plus the
  postal code (branches of one chain share a name and often a street, but rarely
  these numbers).
- Other:
  - Blocking context (10): rank and relative score from both directions, candidate
    counts, rank and gap of a combined similarity within the Source 1 entity's and
    the Source 2/3 record's candidates.
  - Coherence (3): similarity of a candidate to its Source 1 entity's best other
    candidate; records of one business resemble each other, distractors do not.
  - Full-set competition (9): the filter score and rank/gap/count features over
    the full blocked set. Computed only on the filtered set, they make wrong pairs
    look uncontested; on synthetic data that tripled false matches.
  - Source (S2 or S3).

String similarities use rapidfuzz (C++); TF-IDF uses scikit-learn.

**Model type:** LightGBM gradient-boosted trees (binary): 255 leaves, learning rate
0.08, L1 0.5, L2 5, feature fraction 0.7, bagging 0.8, minimum 300 samples per
leaf, up to 2,000 rounds with early stopping; negatives subsampled to at most 12M.
The candidate filter is a second, smaller LightGBM model. Both are trained from
scratch on the provided data (LightGBM is MIT-licensed).

**Threshold selection method:** direct macro F0.5 optimization on held-out Source 1
entities. A pair is a match when its probability is at least a threshold; each
Source 2/3 record is then assigned only to the Source 1 entity with the highest
probability, since it can belong to at most one. The threshold (grid 0.10–0.95),
an optional relative-to-best rule and the assignment step are searched together;
the final choice is a threshold of 0.625 with assignment on.

**Validation protocol:** 80/20 split by Source 1 entity (fixed seed), so all pairs
of an entity fall on one side; the filter and the matcher use the same split.
Macro F0.5 is computed exactly as specified, per Source 1 entity with singletons
included; blocking and filter misses count as false negatives because the true
match counts come from the ground truth. Every change was first run end to end on
a 300K-entity synthetic dataset with US, India and France records and checked with
`validate_submission.py`.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9511 on held-out training entities (US + India);
  0.932 on the public leaderboard.

| Version | Main change | Blocking recall | Validation F0.5 | Public leaderboard |
|---|---|---|---|---|
| Baseline | Forward top-20 blocking, 37 features | 0.9034 | 0.919 | 0.893 |
| v3 | Two-way blocking; TF-IDF, address-number and coherence features; normalization of spelling variants | 0.9409 | 0.9436 | 0.924 |
| v4 | Learned candidate filter; full-set competition features; reverse top-5 | not recorded | ~0.942 | 0.927 |
| v5 (final) | Normalization of French address and name variants | 0.9490 (0.9446 after filter) | 0.9511 | 0.932 |

- **Common false positives (wrong merges):** based on inspected French candidate
  lists and error counts on synthetic data: different businesses on the same street with generic, overlapping
  names ("Chasse Service" at 146 vs "Chasse Club" at 149 Rue du Collège), which
  depend on the house number being parsed; branches of one chain with identical
  names and nearby addresses; and a singleton Source 1 entity whose closest
  look-alike shares its name. Address-number, coherence and competition features
  target these; on synthetic data they cut false matches from 23 to 8.
- **Common false negatives (missed matches):** 5.1% of true pairs never reach the
  candidate set. The hard cases seen when inspecting records are those whose name
  and address share few rare tokens with their Source 1 entity: names run together
  ("honoreloisirssarlcom"), heavy typos combined with reordered or missing address
  components, and records with an empty address. A further 0.5% of the pairs blocking finds are dropped by the
  filter by design. On the leaderboard, France accounts for most of the remaining
  gap between validation and leaderboard (2.6 points for the baseline, 1.9 for the
  final model), consistent with French variants the rules still miss.

---

## 6. Conclusion

Treating candidate generation as a two-sided, learned problem raised blocking
recall from 0.903 to 0.949 while cutting the candidate set the matcher scores to
5.3 per Source 1 entity, and a precision-oriented, F0.5-tuned LightGBM matcher
took the public leaderboard from 0.893 to 0.932. The main lessons: exploit the
one-owner structure in both blocking and assignment, keep competition information
from the full candidate pool after filtering, and read real records from the
unseen country, where most of the remaining error comes from.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the full pipeline, a `README.md` and a
pinned `requirements.txt`. Entry point, run from the submission root with the data
in `dataset/train` and `dataset/test`:

```bash
pip install -r code/business_entity_resolution/requirements.txt
python -u code/business_entity_resolution/src/run_pipeline.py --mode both
```

This trains the candidate filter and matcher, then writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv` (plus the models
and tuned parameters). It takes 207 minutes on a 4-core, 30GB Kaggle CPU notebook,
with an estimated peak of about 25GB; every stage is cached under `cache/`, and
seeds are fixed.

| File | Role |
|---|---|
| `src/run_pipeline.py` | Entry point; stage orchestration, caching, output writing |
| `src/normalize_fast.py` | Name and address normalization |
| `src/blocking_fast.py` | Two-way IDF token blocking |
| `src/features.py` | Candidate-filter features and the 57 matcher features |
| `src/ml_scorer.py` | Filter and matcher training, F0.5 tuning, selection, prediction |

### B. Additional Results

**Blocking recall by depth (final configuration, training data):**

| k | Forward (best k per Source 1 entity) | Reverse (best k per Source 2/3 record) |
|---|---|---|
| 1 | 0.2444 | 0.9104 |
| 2 | 0.4517 | 0.9299 |
| 3 | 0.6156 | 0.9384 |
| 5 | 0.8048 | 0.9473 |
| 10 | 0.8898 | |

**What did not work:**
- Forward-only blocking with 50 candidates per entity: 110M pairs, out of memory
  on 30GB, and still capped by crowding.
- Cutting the candidate file down with the matcher's own probabilities: replaced,
  because the file must be the set the matcher actually scores.
- Competition features computed only on the filtered candidates: tripled false
  matches on synthetic data.

**How France was handled:** no French training data exists, so the design avoids
anything tied to the training countries: no country-specific parameters, relative
instead of absolute blocking scores, and one set of normalization rules for every
record. The French variants were identified by reading unlabeled test records; no
test labels exist or were used. The final normalization pass raised validation
(about 0.942 to 0.951) and the leaderboard (0.927 to 0.932) by similar amounts, so
it helped every country rather than closing the France gap specifically.
