import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def compute_pair_features(s1_row, cand_row):
    name1 = s1_row.get('name_no_suffix', '') or ''
    name2 = cand_row.get('name_no_suffix', '') or ''
    addr1 = s1_row.get('addr_expanded', '') or ''
    addr2 = cand_row.get('addr_expanded', '') or ''

    features = {}

    features['name_jaro_winkler'] = JaroWinkler.similarity(name1, name2)
    features['name_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0
    features['name_partial'] = fuzz.partial_ratio(name1, name2) / 100.0
    features['name_ratio'] = fuzz.ratio(name1, name2) / 100.0
    features['name_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0

    features['addr_jaro_winkler'] = JaroWinkler.similarity(addr1, addr2)
    features['addr_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    features['addr_partial'] = fuzz.partial_ratio(addr1, addr2) / 100.0
    features['addr_ratio'] = fuzz.ratio(addr1, addr2) / 100.0

    postal1 = s1_row.get('postal_code', '')
    postal2 = cand_row.get('postal_code', '')
    if postal1 and postal2:
        features['postal_match'] = 1.0 if postal1 == postal2 else -1.0
    else:
        features['postal_match'] = 0.0

    house1 = s1_row.get('house_number', '')
    house2 = cand_row.get('house_number', '')
    if house1 and house2:
        features['house_match'] = 1.0 if house1 == house2 else -1.0
    else:
        features['house_match'] = 0.0

    country1 = s1_row.get('country', '')
    country2 = cand_row.get('country', '')
    features['country_match'] = 1.0 if country1 == country2 else 0.0

    features['name_exact_match'] = 1.0 if name1 == name2 and name1 != '' else 0.0

    acr1 = s1_row.get('acronym', '')
    acr2 = cand_row.get('acronym', '')
    features['acronym_match'] = 0.0
    if acr1 and (acr1 == name2 or acr1 == acr2):
        features['acronym_match'] = 1.0
    elif acr2 and acr2 == name1:
        features['acronym_match'] = 1.0

    features['is_s2'] = 1.0 if cand_row.get('entity_id', '').startswith('S2') else 0.0
    features['is_s3'] = 1.0 if cand_row.get('entity_id', '').startswith('S3') else 0.0

    return features


def compute_combined_score(features):
    score = (
        0.35 * features['name_jaro_winkler'] +
        0.15 * features['name_token_set'] +
        0.10 * features['name_partial'] +
        0.10 * features['name_ratio'] +
        0.10 * features['addr_token_set'] +
        0.05 * features['addr_jaro_winkler'] +
        0.05 * features['postal_match'] * 0.5 +
        0.05 * features['country_match'] +
        0.05 * features['name_exact_match']
    )
    return score


def similarity_matcher(candidates_df, s1_norm, s2s3_norm, threshold=0.55):
    s1_dict = {row['entity_id']: row for _, row in s1_norm.iterrows()}
    s2s3_dict = {row['entity_id']: row for _, row in s2s3_norm.iterrows()}

    results = []
    total = len(candidates_df)
    for idx, (_, pair) in enumerate(candidates_df.iterrows()):
        if idx % 50000 == 0 and idx > 0:
            print(f"    Scored {idx}/{total} pairs...")
        s1_id = pair['s1_id']
        cand_id = pair['cand_id']

        s1_row = s1_dict.get(s1_id)
        cand_row = s2s3_dict.get(cand_id)
        if s1_row is None or cand_row is None:
            continue

        feats = compute_pair_features(s1_row, cand_row)
        score = compute_combined_score(feats)
        results.append({
            's1_id': s1_id,
            'cand_id': cand_id,
            'score': score,
            **feats,
        })

    scores_df = pd.DataFrame(results)
    return scores_df


def apply_threshold(scores_df, all_s1_ids, t_match=0.55, t_empty=0.45):
    matches = {}
    for s1_id in all_s1_ids:
        matches[s1_id] = []

    if len(scores_df) == 0:
        return matches

    for s1_id, group in scores_df.groupby('s1_id'):
        best_score = group['score'].max()
        if best_score < t_empty:
            matches[s1_id] = []
        else:
            matched = group[group['score'] >= t_match]['cand_id'].tolist()
            matches[s1_id] = matched

    return matches


def resolve_conflicts(matches, scores_df):
    cand_to_s1 = {}
    for s1_id, cands in matches.items():
        for cand_id in cands:
            row = scores_df[(scores_df['s1_id'] == s1_id) & (scores_df['cand_id'] == cand_id)]
            if len(row) > 0:
                score = row.iloc[0]['score']
            else:
                score = 0
            if cand_id not in cand_to_s1 or score > cand_to_s1[cand_id][1]:
                cand_to_s1[cand_id] = (s1_id, score)

    resolved = {s1_id: [] for s1_id in matches}
    for cand_id, (best_s1, _) in cand_to_s1.items():
        resolved[best_s1].append(cand_id)

    return resolved
