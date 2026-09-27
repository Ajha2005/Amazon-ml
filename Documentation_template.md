# Business Entity Resolution: Methodology

## 1. Summary

A five-stage pipeline: normalization, two-way token blocking, a learned candidate
filter, pair features, and a LightGBM matcher with a selection rule tuned for
macro F0.5. Only the provided training data is used: no external data, APIs or
pretrained models. Every dependency is open source (LightGBM and rapidfuzz: MIT;
pandas, NumPy, SciPy, scikit-learn: BSD).

| Version | Main change | Blocking recall (train) | Validation macro F0.5 | Public leaderboard |
|---|---|---|---|---|
| Baseline | Top-20 Source 2/3 records per Source 1 entity, 37 features | 0.9034 | 0.919 | 0.893 |
| v3 | Two-way blocking; TF-IDF, address-number and coherence features; normalization of spelling variants | 0.9409 | 0.9436 | 0.924 |
| v4 | Learned candidate filter; competition features over the full blocked set; wider reverse blocking | TODO | ~0.942 | 0.927 |
| v5 (final) | Normalization of French address and name variants | TODO | TODO | TODO |

Final candidate set: **TODO candidates per Source 1 entity** in `candidate_pairs.tsv`,
which is exactly the set the matcher scores.

## 2. Data observations that shaped the design

- Training: 2.21M Source 1 entities, 10.32M Source 2+3 records, 7.64M true pairs
  (3.46 per Source 1 entity); 5.58% of Source 1 entities have no match.
- Test: 1.73M Source 1 entities, 9.97M Source 2+3 records, including France
  (259K Source 1, 1.43M Source 2+3), which does not appear in training.
- Source 1 is deduplicated, so every Source 2/3 record belongs to at most one
  Source 1 entity. This drives both the blocking (section 4) and the final
  assignment rule (section 5).
- Inspecting unlabeled French test records showed: no postal codes at all;
  house numbers written as "No. 146", "Nº 51", "0021" or placed after the city;
  the region included, omitted or swapped for the department
  ("Hauts-de-France" / "Nord"); legal forms before or inside the name
  ("SARL Dupont", "AS SARL Sportive"); "(France)" inserted into names.

## 3. Normalization (`normalize_fast.py`)

All rules are applied identically to every record regardless of country; the
country label is used only to partition blocking.

- Lower-case, NFKD accent stripping, ligatures (œ → oe), apostrophes joined
  (L'Amitié → lamitie), dotted and spelled-out legal forms collapsed
  (S.A.R.L., s a r l → sarl).
- Names: legal forms and courtesy prefixes stripped from the start
  (SARL, Sté, Ets, M/s), end (repeatedly: "ABC Technologies Pvt Ltd" → "abc")
  and middle; & / and / et unified; transliteration variants (Shri / Sri / Shree);
  common abbreviations (Bros, Mfg, Intl, Ctr); "X trading as Y" / "dba" reduced
  to the trade name; "(France)" and a trailing country word removed. The full
  cleaned name is kept separately for features.
- Addresses: one-pass expansion of street types and landmarks in English, French
  and Indian usage (St, Rd, Bd, R., Crs, Imp., Opp, Ngr, Sec, Extn) and N/S/E/W
  directionals; comma-separated components that are only a region, state or
  country are dropped (a postal code in the component is kept); number prefixes
  (No., Nº, N°) and leading zeros removed; digits split from letters (12bis).
- Extracted fields: postal code (5–6 digits), house number (first token, or the
  number directly before a street word when the address starts with a city or
  region), acronym, legal form.

## 4. Candidate generation

### 4.1 Two-way token blocking (`blocking_fast.py`)

Each record is represented by sparse binary vectors over separate vocabularies
for name words, address words, postal code, name and address word bigrams, name
character trigrams, consonant-skeleton phonetic keys, and name-word@postal
composites. Tokens found in more than 2,000 Source 2/3 records are dropped and
the rest are weighted by IDF, so a pair's blocking score is the IDF mass of the
tokens it shares. Scores come from a chunked sparse matrix product run per
country on 4 processes. No full pairwise comparison is ever made: cost grows
with the postings of shared rare tokens.

Two directions are combined:

- the top 10 Source 2/3 records for each Source 1 entity, and
- the top 5 Source 1 entities for each Source 2/3 record.

The reverse direction exploits the one-owner constraint. A Source 2/3 record's
true Source 1 entity is almost always among its best few, even when that entity
is crowded out of its own top-k by look-alikes (chains, generic names, many
businesses on one street). On training data (v3, reverse top-3), reverse top-1
alone reached 0.908 recall versus 0.888 for forward top-10, and the union reached
0.9409, against 0.9034 for forward top-20 alone.

### 4.2 Learned candidate filter (`features.py`, `ml_scorer.py`)

A LightGBM model (63 leaves) scores every blocked pair using 18 cheap, scale-free
features: relative blocking score and rank from both directions, candidate counts,
name token-set and Jaro-Winkler similarity, address token-set similarity,
postal and house-number equality, the source, and each pair's rank and gap among
its Source 1 entity's and its Source 2/3 record's candidates. The raw blocking
score is excluded because its scale depends on how many records a country has.

The filter is trained on 80% of training Source 1 entities. Its threshold is the
0.5th percentile of true-pair scores on the held-out 20%, so it keeps 99.5% of
the true pairs that blocking found. Blocked pairs go from TODO to **TODO per
Source 1 entity** (on 300K-entity synthetic data: 26.2 to 3.7). The filtered set
is written to `candidate_pairs.tsv` and is exactly what the matcher scores.

## 5. Matcher (`features.py`, `ml_scorer.py`)

### 5.1 Features (57)

- String similarity (10): ratio, token-set, token-sort, partial ratio and
  Jaro-Winkler on the core name; token-set on the full name; ratio, token-set,
  partial ratio and Jaro-Winkler on the address (rapidfuzz, C++).
- TF-IDF cosine (3): name words, name character 3-grams, address words, all
  sublinear TF-IDF, so rare shared words count for more than generic ones.
- Structure (19): exact name match, first/last token match, name lengths and
  token counts, acronym matches, legal-form match, postal and house-number
  equality and presence, address presence, source.
- Address numbers (3): shared count, Jaccard and joint presence of every number
  in the address plus the postal code. Branches of one chain share a name and
  often a street, but rarely these numbers.
- Blocking context (10): blocking rank and relative score from both directions,
  candidate counts, and rank and gap of a combined similarity within the Source 1
  entity's and the Source 2/3 record's candidates.
- Coherence (3): similarity of a candidate to its Source 1 entity's best other
  candidate. Records of one business resemble each other; distractors do not.
- Full-set competition (9): the filter score and the rank/gap/count features
  computed over the full blocked set. Recomputed only on the filtered set, they
  make wrong pairs look uncontested; on synthetic data this tripled false matches.

### 5.2 Model and selection

LightGBM binary classifier: 255 leaves, learning rate 0.08, L1 0.5, L2 5,
feature fraction 0.7, bagging 0.8, minimum 300 samples per leaf, up to 2,000
rounds with early stopping; negatives subsampled to at most 12M.

A pair is predicted as a match when its probability is at least a threshold t.
Each Source 2/3 record is then assigned only to the Source 1 entity with the
highest probability, because it can belong to at most one. The threshold
(grid 0.10–0.95), an optional relative-to-best rule and the assignment step are
tuned together to maximize macro F0.5 on held-out Source 1 entities; the chosen
threshold is around 0.8, reflecting F0.5's weight on precision.

## 6. Validation protocol

- The split is by Source 1 entity (80/20, fixed seed), so all pairs of an entity
  fall on one side; the filter and the matcher use the same split.
- Macro F0.5 is computed exactly as specified: per Source 1 entity, averaged
  over all held-out entities, singletons included. Blocking and filter misses
  count as false negatives because the true match counts come from the ground
  truth, not from the candidate set.
- Each change was first run end to end on a 300K-entity synthetic dataset with
  US, India and France records, and checked with `validate_submission.py`.

## 7. Generalizing to France

With no French training data, the design avoids anything tied to the training
countries: no country-specific parameters, relative instead of absolute
blocking scores, and one set of normalization rules for every record. French
spelling variants were identified by inspecting unlabeled test records
(section 2); no test labels exist or were used. The gap between validation
(US + India) and the leaderboard, most of it attributable to France, shrank from
2.6 points (baseline) to 2.0 (v3) and TODO (v5).

## 8. Scale and reproducibility

- Full run on a 4-core, 30GB Kaggle CPU notebook: about 3–3.5 hours for training
  and test together; estimated peak memory about 25GB.
- Every stage is cached, so a rerun resumes from the last finished stage.
- Seeds are fixed for the split, negative sampling and LightGBM.
- Command: `python -u code/business_entity_resolution/src/run_pipeline.py --mode both`
  (see `code/business_entity_resolution/README.md`).

## 9. What did not work

- Forward-only blocking with 50 candidates per Source 1 entity: 110M pairs,
  out of memory on 30GB, and still capped by crowding.
- Cutting the candidate file down with the matcher's own probabilities:
  replaced, because the file must be the set the matcher actually scores.
- Competition features computed only on filtered candidates: tripled false
  matches on synthetic data; they are now computed on the full blocked set.
