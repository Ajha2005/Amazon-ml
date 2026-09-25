#!/usr/bin/env python3
"""
Day 1 Pipeline: EDA + TF-IDF Blocking + Similarity Threshold Matcher
Produces output/matching_results.tsv and output/candidate_pairs.tsv
"""
import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_dataframe
from blocking import get_candidates
from matcher import similarity_matcher, apply_threshold, resolve_conflicts
from scorer import evaluate, blocking_recall

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def load_data(data_dir, prefix):
    s1 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source1.tsv'), sep='\t')
    s2 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source2.tsv'), sep='\t')
    s3 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source3.tsv'), sep='\t')
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    return s1, s2, s3, s2s3


def write_matching_results(matches, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in sorted(matches.keys()):
            cands = matches[s1_id]
            cand_str = ','.join(sorted(set(cands))) if cands else ''
            f.write(f'{s1_id}\t{cand_str}\n')
    print(f"  Wrote {output_path}")


def write_candidate_pairs(candidates_df, all_s1_ids, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cand_dict = {}
    for s1_id in all_s1_ids:
        cand_dict[s1_id] = []
    for _, row in candidates_df.iterrows():
        cand_dict[row['s1_id']].append(row['cand_id'])

    with open(output_path, 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in sorted(cand_dict.keys()):
            cands = cand_dict[s1_id]
            cand_str = ','.join(sorted(set(cands))) if cands else ''
            f.write(f'{s1_id}\t{cand_str}\n')
    print(f"  Wrote {output_path}")


def tune_thresholds(scores_df, gt_df, all_s1_ids):
    from scorer import f05_per_entity

    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or str(matched).strip() == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matched).split(','))

    best_f05 = 0
    best_t_match = 0.5
    best_t_empty = 0.4

    for t_match in np.arange(0.35, 0.75, 0.05):
        for t_empty in np.arange(max(0.2, t_match - 0.2), t_match + 0.01, 0.05):
            matches = apply_threshold(scores_df, all_s1_ids, t_match=t_match, t_empty=t_empty)
            matches = resolve_conflicts(matches, scores_df)

            entity_scores = []
            for s1_id in gt_dict:
                pred = set(matches.get(s1_id, []))
                true = gt_dict[s1_id]
                entity_scores.append(f05_per_entity(pred, true))

            f05 = np.mean(entity_scores)
            if f05 > best_f05:
                best_f05 = f05
                best_t_match = t_match
                best_t_empty = t_empty

    print(f"  Best thresholds: t_match={best_t_match:.2f}, t_empty={best_t_empty:.2f}, F0.5={best_f05:.4f}")
    return best_t_match, best_t_empty, best_f05


def run_pipeline(mode='train', top_k=30):
    start = time.time()

    if mode == 'train':
        print("\n[1/6] Loading training data...")
        s1, s2, s3, s2s3 = load_data(TRAIN_DIR, 'train')
        gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t')
    else:
        print("\n[1/6] Loading test data...")
        s1, s2, s3, s2s3 = load_data(TEST_DIR, 'test')
        gt = None

    print(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}, S2+S3: {len(s2s3)}")

    print("\n[2/6] Normalizing records...")
    s1_norm = normalize_dataframe(s1)
    s2s3_norm = normalize_dataframe(s2s3)
    print(f"  Normalized {len(s1_norm)} S1 and {len(s2s3_norm)} S2/S3 records")

    print("\n[3/6] Generating candidates (blocking)...")
    candidates = get_candidates(s1_norm, s2s3_norm, top_k=top_k)

    if gt is not None:
        print("\n  Measuring blocking recall...")
        blocking_recall(candidates, gt)

    print("\n[4/6] Computing similarity scores...")
    scores_df = similarity_matcher(candidates, s1_norm, s2s3_norm)
    print(f"  Scored {len(scores_df)} pairs")

    all_s1_ids = s1['entity_id'].tolist()

    if gt is not None:
        print("\n[5/6] Tuning thresholds on training data...")
        t_match, t_empty, val_f05 = tune_thresholds(scores_df, gt, all_s1_ids)
    else:
        t_match, t_empty = 0.55, 0.45
        print(f"\n[5/6] Using default thresholds: t_match={t_match}, t_empty={t_empty}")

    print("\n[6/6] Applying thresholds and conflict resolution...")
    matches = apply_threshold(scores_df, all_s1_ids, t_match=t_match, t_empty=t_empty)
    matches = resolve_conflicts(matches, scores_df)

    if gt is not None:
        print("\n  Final validation score...")
        pred_rows = []
        for s1_id in all_s1_ids:
            cands = matches.get(s1_id, [])
            pred_rows.append({
                'source1_entity_id': s1_id,
                'matched_entity_ids': ','.join(sorted(set(cands))) if cands else ''
            })
        pred_df = pd.DataFrame(pred_rows)
        macro_f05, _ = evaluate(pred_df, gt)
        print(f"  Macro F0.5 on training data: {macro_f05:.4f}")

    matching_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    candidate_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

    if mode == 'test':
        write_matching_results(matches, matching_path)
        write_candidate_pairs(candidates, all_s1_ids, candidate_path)

    elapsed = time.time() - start
    print(f"\nPipeline completed in {elapsed:.1f}s")

    return matches, candidates, scores_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both')
    parser.add_argument('--top-k', type=int, default=30)
    args = parser.parse_args()

    if args.mode in ('train', 'both'):
        print("=" * 60)
        print("TRAINING / VALIDATION RUN")
        print("=" * 60)
        run_pipeline(mode='train', top_k=args.top_k)

    if args.mode in ('test', 'both'):
        print("\n" + "=" * 60)
        print("TEST SUBMISSION RUN")
        print("=" * 60)
        run_pipeline(mode='test', top_k=args.top_k)
