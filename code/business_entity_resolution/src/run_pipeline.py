#!/usr/bin/env python3
"""
End-to-end entity resolution pipeline.

  train: normalize -> top-K blocking -> features -> LightGBM (S1-level holdout)
         -> tune threshold on holdout -> save model + selection params
  test:  normalize -> top-K blocking -> features -> load model -> select matches
         -> output/matching_results.tsv + output/candidate_pairs.tsv

Every stage is cached under cache/ so reruns skip finished work.
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize_fast import normalize_in_chunks_fast
from blocking_fast import get_candidates_topk
from features import compute_features, BACKEND, FEATURE_COLS
from ml_scorer import (ground_truth_index, pair_labels, blocking_report,
                       train_and_tune, predict, select_matches)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
DATA_DIR = {'train': os.path.join(BASE_DIR, 'dataset', 'train'),
            'test': os.path.join(BASE_DIR, 'dataset', 'test')}
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.txt')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'selection_params.json')

NEEDED_COLS = ['entity_id', 'business_name', 'business_address', 'country']
KEEP_NORM = ['entity_id', 'name_clean', 'name_no_suffix', 'legal_suffix', 'acronym',
             'addr_expanded', 'postal_code', 'house_number', 'country']


def _read_tsv(path):
    header = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
    cols = [c for c in NEEDED_COLS if c in header]
    return pd.read_csv(path, sep='\t', usecols=cols, dtype=str, na_filter=False)


def _cached(path, build):
    if os.path.exists(path):
        t = time.time()
        obj = pd.read_pickle(path)
        print(f"  Loaded cache {os.path.basename(path)} ({time.time() - t:.0f}s)")
        return obj
    obj = build()
    t = time.time()
    pd.to_pickle(obj, path)
    print(f"  Cached -> {os.path.basename(path)} ({time.time() - t:.0f}s)")
    return obj


def load_normalized(mode):
    def build():
        d = DATA_DIR[mode]
        frames = {}
        for src in ('source1', 'source2', 'source3'):
            frames[src] = _read_tsv(os.path.join(d, f'{mode}_{src}.tsv'))
            print(f"  {mode}_{src}.tsv: {len(frames[src]):,} rows")
        t = time.time()
        s1 = normalize_in_chunks_fast(frames['source1'])[KEEP_NORM]
        c = normalize_in_chunks_fast(
            pd.concat([frames['source2'], frames['source3']], ignore_index=True))[KEEP_NORM]
        c['src'] = np.r_[np.zeros(len(frames['source2']), np.int8),
                         np.ones(len(frames['source3']), np.int8)]
        print(f"  Normalized in {time.time() - t:.0f}s")
        return s1, c
    return _cached(os.path.join(CACHE_DIR, f'{mode}_norm.pkl'), build)


def build_pairs(mode, k, df_cap):
    print(f"\n[1/4] Loading + normalizing {mode} data...")
    s1, c = load_normalized(mode)
    print(f"  S1={len(s1):,}  S2+S3={len(c):,}")

    print(f"\n[2/4] Blocking (top-{k} per S1, df_cap={df_cap})...")
    cands = _cached(os.path.join(CACHE_DIR, f'{mode}_cands_k{k}_cap{df_cap}.pkl'),
                    lambda: get_candidates_topk(s1, c, k=k, df_cap=df_cap))

    print(f"\n[3/4] Features...")
    feats = _cached(os.path.join(CACHE_DIR, f'{mode}_feats_k{k}_cap{df_cap}_{BACKEND}.pkl'),
                    lambda: compute_features(cands.copy(), s1, c))
    del cands
    s1_ids = s1['entity_id'].to_numpy(dtype=object)
    c_ids = c['entity_id'].to_numpy(dtype=object)
    del s1, c
    gc.collect()
    return feats, s1_ids, c_ids


def run_train(k, df_cap):
    feats, s1_ids, c_ids = build_pairs('train', k, df_cap)
    gt = pd.read_csv(os.path.join(DATA_DIR['train'], 'train_ground_truth.tsv'),
                     sep='\t', dtype=str, na_filter=False)
    true_count, true_keys = ground_truth_index(gt, s1_ids, c_ids)
    labels = pair_labels(feats, true_keys, len(c_ids))
    del gt, true_keys
    print(f"  S1 with no true match (singletons): {(true_count == 0).mean():.2%}")
    blocking_report(feats, labels, true_count)

    print(f"\n[4/4] Training LightGBM...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_and_tune(feats, labels, true_count, len(s1_ids), MODEL_PATH, PARAMS_PATH,
                   backend=BACKEND)


def _write_grouped(path, col, s1_ids, c_ids, s1_idx, c_idx):
    order = np.lexsort((c_idx, s1_idx))
    df = pd.DataFrame({'s': s1_idx[order], 'c': c_ids[c_idx[order]]})
    joined = df.groupby('s', sort=False)['c'].agg(','.join)
    values = np.full(len(s1_ids), '', dtype=object)
    values[joined.index.to_numpy()] = joined.to_numpy(dtype=object)
    pd.DataFrame({'source1_entity_id': s1_ids, col: values}).to_csv(path, sep='\t', index=False)
    print(f"  Wrote {path}")


def run_test(k, df_cap):
    import lightgbm as lgb
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    if params.get('backend') != BACKEND:
        print(f"  WARNING: model trained with {params.get('backend')} features, "
              f"now using {BACKEND} — scores may be off")
    if params.get('features') != FEATURE_COLS:
        sys.exit("Feature list changed since training — retrain with --mode train")
    booster = lgb.Booster(model_file=MODEL_PATH)
    print(f"  Loaded model + params: {params}")

    feats, s1_ids, c_ids = build_pairs('test', k, df_cap)

    print(f"\n[4/4] Scoring + writing output...")
    prob = predict(booster, feats)
    s1_idx = feats['s1_idx'].to_numpy()
    c_idx = feats['c_idx'].to_numpy()
    s1_max = pd.Series(prob).groupby(s1_idx).transform('max').to_numpy()
    sel = select_matches(s1_idx, c_idx, prob, params['threshold'], params['rel'],
                         s1_max, params['resolve'])
    n_matched = len(np.unique(s1_idx[sel]))
    print(f"  {len(sel):,} matched pairs; {n_matched:,}/{len(s1_ids):,} S1 records have >=1 match")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _write_grouped(os.path.join(OUTPUT_DIR, 'matching_results.tsv'), 'matched_entity_ids',
                   s1_ids, c_ids, s1_idx[sel], c_idx[sel])
    _write_grouped(os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv'), 'candidate_entity_ids',
                   s1_ids, c_ids, s1_idx, c_idx)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'test', 'both'], default='both')
    ap.add_argument('--k', type=int, default=20, help='candidates kept per S1 record')
    ap.add_argument('--df-cap', type=int, default=1000,
                    help='drop blocking tokens found in more S2/S3 records than this')
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    start = time.time()
    if args.mode in ('train', 'both'):
        run_train(args.k, args.df_cap)
        gc.collect()
    if args.mode in ('test', 'both'):
        run_test(args.k, args.df_cap)
    print(f"\nTotal time: {(time.time() - start) / 60:.1f} min")
