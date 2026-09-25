import pandas as pd
import numpy as np


def f05_per_entity(predicted_set, true_set):
    if len(true_set) == 0 and len(predicted_set) == 0:
        return 1.0
    if len(true_set) == 0 and len(predicted_set) > 0:
        return 0.0
    if len(predicted_set) == 0 and len(true_set) > 0:
        return 0.0

    tp = len(predicted_set & true_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta = 0.5
    f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return f05


def evaluate(predictions_df, ground_truth_df):
    gt_dict = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matched).split(','))

    pred_dict = {}
    for _, row in predictions_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            pred_dict[s1_id] = set()
        else:
            pred_dict[s1_id] = set(str(matched).split(','))

    scores = []
    for s1_id in gt_dict:
        pred = pred_dict.get(s1_id, set())
        true = gt_dict[s1_id]
        scores.append(f05_per_entity(pred, true))

    macro_f05 = np.mean(scores)
    return macro_f05, scores


def blocking_recall(candidates_df, ground_truth_df):
    gt_dict = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matched).split(','))

    cand_dict = {}
    for _, row in candidates_df.iterrows():
        s1_id = row['s1_id']
        cand_id = row['cand_id']
        if s1_id not in cand_dict:
            cand_dict[s1_id] = set()
        cand_dict[s1_id].add(cand_id)

    total_true = 0
    found = 0
    for s1_id, true_matches in gt_dict.items():
        if len(true_matches) == 0:
            continue
        cands = cand_dict.get(s1_id, set())
        total_true += len(true_matches)
        found += len(true_matches & cands)

    recall = found / total_true if total_true > 0 else 1.0
    print(f"  Blocking recall: {found}/{total_true} = {recall:.4f}")
    return recall
