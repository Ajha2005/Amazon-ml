#!/usr/bin/env python3
"""
End-to-end entity resolution pipeline.

  1. normalize names and addresses
  2. token blocking: top-k S2/S3 records per S1 + top-kc S1 records per S2/S3
  3. candidate filter: a small LightGBM on cheap features drops clearly wrong pairs
  4. full features for the filtered pairs only
  5. LightGBM matcher (S1-level holdout, tuned selection) -> matching_results.tsv

candidate_pairs.tsv is exactly the filtered set from step 3, i.e. every pair the
matcher scores.

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
from features import (compute_features, prefilter_features, BACKEND, FEATURE_COLS,
                      PREFILTER_COLS, CARRIED_COLS)
from ml_scorer import (ground_truth_index, pair_labels, blocking_report,
                       train_and_tune, train_prefilter, predict, select_matches)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
DATA_DIR = {'train': os.path.join(BASE_DIR, 'dataset', 'train'),
            'test': os.path.join(BASE_DIR, 'dataset', 'test')}
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.txt')
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'selection_params.json')
FILTER_MODEL_PATH = os.path.join(OUTPUT_DIR, 'filter_model.txt')
FILTER_PARAMS_PATH = os.path.join(OUTPUT_DIR, 'filter_params.json')

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
    return _cached(os.path.join(CACHE_DIR, f'{mode}_norm_n4.pkl'), build)


def stage1(mode, k, kc, df_cap):
    """Normalization, token blocking and the cheap filter features for every blocking pair."""
    print(f"\n[1/5] Loading + normalizing {mode} data...")
    s1, c = load_normalized(mode)
    print(f"  S1={len(s1):,}  S2+S3={len(c):,}")

    tag = f'{mode}_n4_k{k}_kc{kc}_cap{df_cap}'
    print(f"\n[2/5] Blocking (top-{k} per S1 + top-{kc} per S2/S3, df_cap={df_cap})...")
    cands = _cached(os.path.join(CACHE_DIR, f'{tag}_cands.pkl'),
                    lambda: get_candidates_topk(s1, c, k=k, kc=kc, df_cap=df_cap))
    print(f"\n[3/5] Candidate filter features...")
    pf = _cached(os.path.join(CACHE_DIR, f'{tag}_filterfeats2_{BACKEND}.pkl'),
                 lambda: prefilter_features(cands, s1, c))
    return s1, c, cands, pf, tag


def stage2(tag, threshold, s1, c, cands, pf, score, keep):
    """Full features for the pairs the filter kept, plus full-set competition
    features and the filter score."""
    kept = cands.loc[keep, ['s1_idx', 'c_idx', 'block_score', 'block_rank', 'block_rank_c']]
    kept = kept.reset_index(drop=True)
    print(f"\n[4/5] Full features for the {len(kept):,} filtered candidates...")

    def build():
        feats = compute_features(kept, s1, c)
        for src, dst in CARRIED_COLS.items():
            feats[dst] = pf[src].to_numpy()[keep]
        feats['filter_score'] = score[keep]
        return feats
    return _cached(os.path.join(CACHE_DIR, f'{tag}_f{threshold:.6f}_feats_v7_{BACKEND}.pkl'), build)


def run_train(k, kc, df_cap, filter_recall):
    s1, c, cands, pf, tag = stage1('train', k, kc, df_cap)
    s1_ids = s1['entity_id'].to_numpy(dtype=object)
    c_ids = c['entity_id'].to_numpy(dtype=object)
    gt = pd.read_csv(os.path.join(DATA_DIR['train'], 'train_ground_truth.tsv'),
                     sep='\t', dtype=str, na_filter=False)
    true_count, true_keys = ground_truth_index(gt, s1_ids, c_ids)
    labels_all = pair_labels(cands, true_keys, len(c_ids))
    del gt, true_keys
    print(f"  S1 with no true match (singletons): {(true_count == 0).mean():.2%}")
    blocking_report(cands, labels_all, true_count)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Training candidate filter (target: keep {filter_recall:.1%} of true pairs)...")
    score, threshold = train_prefilter(pf, PREFILTER_COLS, cands['s1_idx'].to_numpy(), labels_all,
                                       len(s1_ids), true_count, filter_recall, FILTER_MODEL_PATH)
    with open(FILTER_PARAMS_PATH, 'w') as f:
        json.dump({'threshold': threshold, 'target_recall': filter_recall,
                   'features': PREFILTER_COLS, 'k': k, 'kc': kc, 'df_cap': df_cap}, f, indent=2)
    keep = score >= threshold
    feats = stage2(tag, threshold, s1, c, cands, pf, score, keep)
    labels = labels_all[keep]
    del s1, c, cands, pf, score, labels_all
    gc.collect()

    print(f"\n[5/5] Training LightGBM matcher...")
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


def run_test(k, kc, df_cap):
    import lightgbm as lgb
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    with open(FILTER_PARAMS_PATH) as f:
        fparams = json.load(f)
    if params.get('backend') != BACKEND:
        print(f"  WARNING: model trained with {params.get('backend')} features, "
              f"now using {BACKEND} — scores may be off")
    if params.get('features') != FEATURE_COLS or fparams.get('features') != PREFILTER_COLS:
        sys.exit("Feature list changed since training — retrain with --mode train")
    if (fparams['k'], fparams['kc'], fparams['df_cap']) != (k, kc, df_cap):
        sys.exit(f"Blocking settings differ from training {fparams} — use the same --k/--kc/--df-cap")
    booster = lgb.Booster(model_file=MODEL_PATH)
    filter_booster = lgb.Booster(model_file=FILTER_MODEL_PATH)
    print(f"  Loaded matcher params: { {x: params[x] for x in ('threshold', 'rel', 'resolve', 'val_f05')} }")
    print(f"  Loaded filter threshold: {fparams['threshold']:.6f}")

    s1, c, cands, pf, tag = stage1('test', k, kc, df_cap)
    s1_ids = s1['entity_id'].to_numpy(dtype=object)
    c_ids = c['entity_id'].to_numpy(dtype=object)
    score = predict(filter_booster, pf, cols=PREFILTER_COLS)
    keep = score >= fparams['threshold']
    print(f"  Filter: {len(cands):,} -> {int(keep.sum()):,} pairs "
          f"({len(cands) / len(s1_ids):.1f} -> {keep.sum() / len(s1_ids):.1f} per S1)")
    feats = stage2(tag, fparams['threshold'], s1, c, cands, pf, score, keep)
    del s1, c, cands, pf, score
    gc.collect()

    print(f"\n[5/5] Scoring + writing output...")
    prob = predict(booster, feats)
    s1_idx = feats['s1_idx'].to_numpy()
    c_idx = feats['c_idx'].to_numpy()
    s1_max = pd.Series(prob).groupby(s1_idx).transform('max').to_numpy()
    sel = select_matches(s1_idx, c_idx, prob, params['threshold'], params['rel'],
                         s1_max, params['resolve'])
    n_matched = len(np.unique(s1_idx[sel]))
    print(f"  {len(sel):,} matched pairs; {n_matched:,}/{len(s1_ids):,} S1 records have >=1 match")
    print(f"  candidate_pairs: {len(feats):,} pairs = exactly the set the matcher scored "
          f"({len(feats) / len(s1_ids):.1f} per S1)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _write_grouped(os.path.join(OUTPUT_DIR, 'matching_results.tsv'), 'matched_entity_ids',
                   s1_ids, c_ids, s1_idx[sel], c_idx[sel])
    _write_grouped(os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv'), 'candidate_entity_ids',
                   s1_ids, c_ids, s1_idx, c_idx)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'test', 'both'], default='both')
    ap.add_argument('--k', type=int, default=10, help='candidates kept per S1 record')
    ap.add_argument('--kc', type=int, default=5, help='S1 candidates kept per S2/S3 record')
    ap.add_argument('--df-cap', type=int, default=2000,
                    help='drop blocking tokens found in more S2/S3 records than this')
    ap.add_argument('--filter-recall', type=float, default=0.995,
                    help='share of blocking\'s true pairs the candidate filter must keep')
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    start = time.time()
    if args.mode in ('train', 'both'):
        run_train(args.k, args.kc, args.df_cap, args.filter_recall)
        gc.collect()
    if args.mode in ('test', 'both'):
        run_test(args.k, args.kc, args.df_cap)
    print(f"\nTotal time: {(time.time() - start) / 60:.1f} min")
