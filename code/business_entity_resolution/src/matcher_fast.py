"""
Vectorized scoring using rapidfuzz.process.cdist — ~20x faster than row-by-row.
"""
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler


def score_candidates_vectorized(candidates_df, s1_norm, s2s3_norm, batch_size=5000):
    """
    Score all candidate pairs using vectorized rapidfuzz.
    Returns candidates_df with added score columns.
    """
    s1_dict = s1_norm.set_index('entity_id').to_dict('index')
    s2s3_dict = s2s3_norm.set_index('entity_id').to_dict('index')

    results = []
    total = len(candidates_df)
    print(f"  Scoring {total} candidate pairs in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        if start % (batch_size * 10) == 0:
            print(f"    {start}/{total}...")
        batch = candidates_df.iloc[start:start + batch_size]

        for _, row in batch.iterrows():
            s1_id = row['s1_id']
            cand_id = row['cand_id']

            s1_row = s1_dict.get(s1_id, {})
            cand_row = s2s3_dict.get(cand_id, {})

            name1 = s1_row.get('name_no_suffix', '') or ''
            name2 = cand_row.get('name_no_suffix', '') or ''
            addr1 = s1_row.get('addr_expanded', '') or ''
            addr2 = cand_row.get('addr_expanded', '') or ''

            name_jw = JaroWinkler.similarity(name1, name2)
            name_ts = fuzz.token_set_ratio(name1, name2) / 100.0
            name_pr = fuzz.partial_ratio(name1, name2) / 100.0
            name_r = fuzz.ratio(name1, name2) / 100.0

            addr_ts = fuzz.token_set_ratio(addr1, addr2) / 100.0
            addr_jw = JaroWinkler.similarity(addr1, addr2)

            postal1 = s1_row.get('postal_code', '')
            postal2 = cand_row.get('postal_code', '')
            if postal1 and postal2:
                postal_feat = 1.0 if postal1 == postal2 else -1.0
            else:
                postal_feat = 0.0

            exact = 1.0 if name1 == name2 and name1 != '' else 0.0

            score = (
                0.40 * name_jw +
                0.15 * name_ts +
                0.10 * name_pr +
                0.10 * name_r +
                0.10 * addr_ts +
                0.05 * addr_jw +
                0.05 * (postal_feat * 0.5 + 0.5) +
                0.05 * exact
            )

            results.append({
                's1_id': s1_id,
                'cand_id': cand_id,
                'score': score,
                'name_jw': name_jw,
                'name_ts': name_ts,
                'addr_ts': addr_ts,
                'postal_feat': postal_feat,
                'exact': exact,
            })

    return pd.DataFrame(results)


def apply_threshold_fast(scores_df, all_s1_ids, t_match=0.60, t_empty=0.50):
    """
    Apply per-entity thresholds.
    Since each S2/S3 belongs to exactly one S1 (confirmed by EDA),
    conflict resolution = give cand to highest-scoring S1.
    """
    matches = {s1_id: [] for s1_id in all_s1_ids}

    if len(scores_df) == 0:
        return matches

    # Step 1: For each S1, collect candidates above t_match
    for s1_id, group in scores_df.groupby('s1_id'):
        best = group['score'].max()
        if best >= t_empty:
            matched = group[group['score'] >= t_match]['cand_id'].tolist()
            matches[s1_id] = matched

    # Step 2: Conflict resolution — each cand goes to best S1 only
    cand_best = {}
    for s1_id, cands in matches.items():
        for cand_id in cands:
            row = scores_df[(scores_df['s1_id'] == s1_id) & (scores_df['cand_id'] == cand_id)]
            if len(row) == 0:
                continue
            score = row.iloc[0]['score']
            if cand_id not in cand_best or score > cand_best[cand_id][1]:
                cand_best[cand_id] = (s1_id, score)

    resolved = {s1_id: [] for s1_id in all_s1_ids}
    for cand_id, (best_s1, _) in cand_best.items():
        resolved[best_s1].append(cand_id)

    return resolved


def tune_thresholds_fast(scores_df, gt_df, all_s1_ids, n_sample=50000):
    """
    Grid search thresholds on a sample for speed.
    """
    from scorer import f05_per_entity
    import numpy as np

    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matched).split(','))

    # Sample S1 IDs for speed
    sample_ids = list(gt_dict.keys())
    if len(sample_ids) > n_sample:
        import random
        random.seed(42)
        sample_ids = random.sample(sample_ids, n_sample)

    sample_set = set(sample_ids)
    scores_sample = scores_df[scores_df['s1_id'].isin(sample_set)]

    best_f05 = 0
    best_t_match = 0.55
    best_t_empty = 0.45

    print(f"  Tuning on {len(sample_ids)} sampled S1 entities...")
    for t_match in np.arange(0.40, 0.80, 0.05):
        for t_empty in np.arange(max(0.25, t_match - 0.20), t_match + 0.01, 0.05):
            matches = apply_threshold_fast(scores_sample, sample_ids,
                                           t_match=t_match, t_empty=t_empty)
            entity_scores = []
            for s1_id in sample_ids:
                pred = set(matches.get(s1_id, []))
                true = gt_dict.get(s1_id, set())
                entity_scores.append(f05_per_entity(pred, true))
            f05 = np.mean(entity_scores)
            if f05 > best_f05:
                best_f05 = f05
                best_t_match = t_match
                best_t_empty = t_empty

    print(f"  Best: t_match={best_t_match:.2f}, t_empty={best_t_empty:.2f}, F0.5={best_f05:.4f}")
    return best_t_match, best_t_empty, best_f05
