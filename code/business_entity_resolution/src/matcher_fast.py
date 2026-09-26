"""
Scoring using only Python built-ins (difflib) — no rapidfuzz required.
"""
import pandas as pd
import numpy as np
from difflib import SequenceMatcher


def _jaro(s1, s2):
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0
    match_dist = max(l1, l2) // 2 - 1
    m1 = [False] * l1
    m2 = [False] * l2
    matches = 0
    for i in range(l1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, l2)
        for j in range(lo, hi):
            if not m2[j] and s1[i] == s2[j]:
                m1[i] = m2[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(l1):
        if not m1[i]:
            continue
        while not m2[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    return (matches / l1 + matches / l2 + (matches - t / 2) / matches) / 3


def _jaro_winkler(s1, s2, p=0.1):
    j = _jaro(s1, s2)
    prefix = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return j + prefix * p * (1 - j)


def _ratio(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()


def _partial_ratio(s1, s2):
    if not s1 or not s2:
        return 0.0
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    best = 0.0
    for i in range(len(s2) - len(s1) + 1):
        r = SequenceMatcher(None, s1, s2[i:i + len(s1)]).ratio()
        if r > best:
            best = r
    return best


def _token_set_ratio(s1, s2):
    t1 = set(s1.split())
    t2 = set(s2.split())
    inter = t1 & t2
    d1 = t1 - t2
    d2 = t2 - t1
    t0 = ' '.join(sorted(inter))
    a = ' '.join(sorted(inter) + sorted(d1))
    b = ' '.join(sorted(inter) + sorted(d2))
    return max(
        SequenceMatcher(None, t0, a).ratio(),
        SequenceMatcher(None, t0, b).ratio(),
        SequenceMatcher(None, a, b).ratio(),
    )


def score_candidates_vectorized(candidates_df, s1_norm, s2s3_norm, batch_size=5000):
    s1_dict = s1_norm.set_index('entity_id').to_dict('index')
    s2s3_dict = s2s3_norm.set_index('entity_id').to_dict('index')

    results = []
    total = len(candidates_df)
    print(f"  Scoring {total:,} candidate pairs...")

    for start in range(0, total, batch_size):
        if start % (batch_size * 20) == 0 and start > 0:
            print(f"    {start:,}/{total:,}...")
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

            name_jw = _jaro_winkler(name1, name2)
            name_ts = _token_set_ratio(name1, name2)
            name_pr = _partial_ratio(name1, name2)
            name_r = _ratio(name1, name2)

            addr_ts = _token_set_ratio(addr1, addr2)
            addr_jw = _jaro_winkler(addr1, addr2)

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
    matches = {s1_id: [] for s1_id in all_s1_ids}

    if len(scores_df) == 0:
        return matches

    for s1_id, group in scores_df.groupby('s1_id'):
        best = group['score'].max()
        if best >= t_empty:
            matched = group[group['score'] >= t_match]['cand_id'].tolist()
            matches[s1_id] = matched

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
    from scorer import f05_per_entity

    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matched).split(','))

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

    print(f"  Tuning on {len(sample_ids):,} sampled S1 entities...")
    for t_match in np.arange(0.40, 0.80, 0.05):
        for t_empty in np.arange(max(0.25, t_match - 0.20), t_match + 0.01, 0.05):
            m = apply_threshold_fast(scores_sample, sample_ids,
                                     t_match=t_match, t_empty=t_empty)
            scores = [f05_per_entity(set(m.get(s, [])), gt_dict.get(s, set()))
                      for s in sample_ids]
            f05 = np.mean(scores)
            if f05 > best_f05:
                best_f05 = f05
                best_t_match = t_match
                best_t_empty = t_empty

    print(f"  Best: t_match={best_t_match:.2f}, t_empty={best_t_empty:.2f}, F0.5={best_f05:.4f}")
    return best_t_match, best_t_empty, best_f05
