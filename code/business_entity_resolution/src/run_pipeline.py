#!/usr/bin/env python3
"""
Scalable pipeline for 2M+ S1 records.
Memory-efficient: loads only needed columns, frees RAM between stages.
"""
import os
import sys
import gc
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from normalize_fast import normalize_in_chunks_fast
from blocking_fast import get_candidates_fast
from matcher_fast import score_candidates_vectorized, apply_threshold_fast, tune_thresholds_fast
from scorer import evaluate, blocking_recall

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Only load the 4 columns the pipeline actually uses
NEEDED_COLS = ['entity_id', 'business_name', 'business_address', 'country']


def _read_tsv(path):
    """Read TSV loading only needed columns to save memory."""
    # Peek at header to find which needed cols exist
    header = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
    cols = [c for c in NEEDED_COLS if c in header]
    return pd.read_csv(path, sep='\t', usecols=cols, dtype=str, na_filter=False)


def load_data(data_dir, prefix):
    print(f"  Loading {prefix}_source1.tsv...")
    s1 = _read_tsv(os.path.join(data_dir, f'{prefix}_source1.tsv'))
    print(f"    {len(s1):,} rows, {s1.memory_usage(deep=True).sum()/1e6:.0f} MB")

    print(f"  Loading {prefix}_source2.tsv...")
    s2 = _read_tsv(os.path.join(data_dir, f'{prefix}_source2.tsv'))
    print(f"    {len(s2):,} rows, {s2.memory_usage(deep=True).sum()/1e6:.0f} MB")

    print(f"  Loading {prefix}_source3.tsv...")
    s3 = _read_tsv(os.path.join(data_dir, f'{prefix}_source3.tsv'))
    print(f"    {len(s3):,} rows, {s3.memory_usage(deep=True).sum()/1e6:.0f} MB")

    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    return s1, s2s3


def write_matching_results(matches, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in sorted(matches.keys()):
            cands = sorted(set(matches[s1_id]))
            f.write(f'{s1_id}\t{",".join(cands)}\n')
    print(f"  Wrote {output_path}")


def write_candidate_pairs(candidates_df, all_s1_ids, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cand_dict = {s1_id: [] for s1_id in all_s1_ids}
    for _, row in candidates_df.iterrows():
        cand_dict[row['s1_id']].append(row['cand_id'])

    with open(output_path, 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in sorted(cand_dict.keys()):
            cands = sorted(set(cand_dict[s1_id]))
            f.write(f'{s1_id}\t{",".join(cands)}\n')
    print(f"  Wrote {output_path}")


def run_pipeline(mode='test', top_k=10, t_match=None, t_empty=None):
    total_start = time.time()

    # --- 1. Load ---
    print("\n[1/6] Loading data (needed columns only)...")
    if mode in ('train', 'validate'):
        s1, s2s3 = load_data(TRAIN_DIR, 'train')
        gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t',
                         dtype=str, na_filter=False)
    else:
        s1, s2s3 = load_data(TEST_DIR, 'test')
        gt = None
    print(f"  S1={len(s1):,}, S2+S3={len(s2s3):,}")

    all_s1_ids = s1['entity_id'].tolist()

    # --- 2. Normalize ---
    print("\n[2/6] Normalizing records (vectorized)...")
    t = time.time()
    print("  Normalizing S1...")
    s1_norm = normalize_in_chunks_fast(s1, chunk_size=500000)
    del s1
    gc.collect()

    print("  Normalizing S2+S3...")
    s2s3_norm = normalize_in_chunks_fast(s2s3, chunk_size=500000)
    del s2s3
    gc.collect()
    print(f"  Done in {time.time()-t:.0f}s")

    # --- 3. Block ---
    print("\n[3/6] Generating candidates...")
    t = time.time()
    candidates = get_candidates_fast(s1_norm, s2s3_norm, top_k=top_k)
    print(f"  Blocking done in {time.time()-t:.0f}s — {len(candidates):,} pairs")

    if gt is not None:
        print("  Measuring blocking recall...")
        blocking_recall(candidates, gt)

    # --- 4. Score ---
    print("\n[4/6] Scoring candidate pairs...")
    t = time.time()
    scores_df = score_candidates_vectorized(candidates, s1_norm, s2s3_norm, batch_size=10000)
    del candidates
    gc.collect()
    print(f"  Scoring done in {time.time()-t:.0f}s")

    # --- 5. Thresholds ---
    if t_match is None or t_empty is None:
        if gt is not None:
            print("\n[5/6] Tuning thresholds...")
            t_match, t_empty, val_f05 = tune_thresholds_fast(scores_df, gt, all_s1_ids)
        else:
            t_match, t_empty = 0.60, 0.50
            print(f"\n[5/6] Using default thresholds: t_match={t_match}, t_empty={t_empty}")
    else:
        print(f"\n[5/6] Using provided thresholds: t_match={t_match}, t_empty={t_empty}")

    # --- 6. Apply & write ---
    print("\n[6/6] Applying thresholds and writing output...")
    matches = apply_threshold_fast(scores_df, all_s1_ids, t_match=t_match, t_empty=t_empty)

    if gt is not None:
        pred_rows = [{'source1_entity_id': sid,
                      'matched_entity_ids': ','.join(sorted(set(matches.get(sid, []))))}
                     for sid in all_s1_ids]
        pred_df = pd.DataFrame(pred_rows)
        macro_f05, _ = evaluate(pred_df, gt)
        print(f"  Macro F0.5 on {mode} data: {macro_f05:.4f}")

    if mode == 'test':
        # Rebuild candidates from scores_df for output
        cands_for_output = scores_df[['s1_id', 'cand_id']].copy()
        write_matching_results(matches, os.path.join(OUTPUT_DIR, 'matching_results.tsv'))
        write_candidate_pairs(cands_for_output, all_s1_ids,
                              os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv'))

    elapsed = time.time() - total_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    return matches, scores_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'validate', 'test'], default='test')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--t-match', type=float, default=None)
    parser.add_argument('--t-empty', type=float, default=None)
    args = parser.parse_args()
    run_pipeline(mode=args.mode, top_k=args.top_k,
                 t_match=args.t_match, t_empty=args.t_empty)
