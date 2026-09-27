"""
LightGBM pair classifier + vectorized match selection and macro F0.5.
"""
import json
import time

import numpy as np
import pandas as pd

from features import FEATURE_COLS


# ── ground truth ──────────────────────────────────────────────────────────────

def ground_truth_index(gt, s1_ids, c_ids):
    """Returns (true_count per S1 index, sorted int64 keys s1_idx * n_c + c_idx of true pairs)."""
    g = gt[['source1_entity_id', 'matched_entity_ids']].fillna('')
    g = g[g['matched_entity_ids'].str.strip() != '']
    ex = g.assign(c=g['matched_entity_ids'].str.split(',')).explode('c')
    s1_pos = pd.Index(s1_ids).get_indexer(ex['source1_entity_id'].to_numpy(dtype=object))
    c_pos = pd.Index(c_ids).get_indexer(ex['c'].str.strip().to_numpy(dtype=object))
    bad = (s1_pos < 0) | (c_pos < 0)
    if bad.any():
        print(f"  WARNING: {bad.sum():,} ground-truth pairs reference unknown IDs")
    true_count = np.bincount(s1_pos[s1_pos >= 0], minlength=len(s1_ids))
    ok = ~bad
    keys = np.unique(s1_pos[ok].astype(np.int64) * len(c_ids) + c_pos[ok])
    return true_count, keys


def pair_labels(cands, true_keys, n_c):
    keys = cands['s1_idx'].to_numpy().astype(np.int64) * n_c + cands['c_idx'].to_numpy()
    return np.isin(keys, true_keys)


def blocking_report(cands, labels, true_count):
    total = max(int(true_count.sum()), 1)
    print(f"  Ground-truth pairs: {total:,}")
    print(f"  RECALL OF ALL {len(cands):,} CANDIDATE PAIRS = {labels.sum() / total:.4f}")
    for col, side in (('block_rank', 'S1-side'), ('block_rank_c', 'S2/S3-side')):
        if col not in cands:
            continue
        rank = cands[col].to_numpy()
        for k in (1, 2, 3, 5, 10, 15, 20, 30, 50):
            if k > rank.max():
                break
            m = rank < k
            print(f"    {side} recall@{k:<3} = {labels[m].sum() / total:.4f}   ({int(m.sum()):,} pairs)")


# ── selection + metric ───────────────────────────────────────────────────────

def select_matches(s1_idx, c_idx, prob, threshold, rel=0.0, s1_max=None, resolve=True):
    """Indices of pairs predicted as matches."""
    keep = prob >= threshold
    if rel > 0:
        keep &= prob >= rel * s1_max
    idx = np.flatnonzero(keep)
    if resolve and len(idx):
        # every S2/S3 record belongs to exactly one S1 record: keep its best S1
        order = idx[np.argsort(-prob[idx], kind='stable')]
        _, first = np.unique(c_idx[order], return_index=True)
        idx = np.sort(order[first])
    return idx


def macro_f05(s1_idx_sel, labels_sel, true_count, s1_mask=None):
    n = len(true_count)
    tp = np.bincount(s1_idx_sel, weights=labels_sel.astype(np.float64), minlength=n)
    pc = np.bincount(s1_idx_sel, minlength=n)
    fp, fn = pc - tp, true_count - tp
    denom = 1.25 * tp + 0.25 * fn + fp
    f = np.where(pc + true_count == 0, 1.0, 1.25 * tp / np.maximum(denom, 1e-12))
    return float(f[s1_mask].mean() if s1_mask is not None else f.mean())


def tune_selection(s1_idx, c_idx, prob, labels, true_count, s1_mask):
    s1_max = pd.Series(prob).groupby(s1_idx).transform('max').to_numpy()
    best = (-1.0, None)
    for resolve in (True, False):
        for rel in (0.0, 0.5):
            for t in np.round(np.arange(0.10, 0.96, 0.025), 3):
                sel = select_matches(s1_idx, c_idx, prob, t, rel, s1_max, resolve)
                f = macro_f05(s1_idx[sel], labels[sel], true_count, s1_mask)
                if f > best[0]:
                    best = (f, {'threshold': float(t), 'rel': rel, 'resolve': resolve})
    print(f"  Best selection: {best[1]}  ->  validation macro F0.5 = {best[0]:.5f}")
    return best[1], best[0]


# ── model ─────────────────────────────────────────────────────────────────────

LGB_PARAMS = dict(
    objective='binary', learning_rate=0.08, num_leaves=255, min_data_in_leaf=300,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.5, lambda_l2=5.0,
    max_bin=511, verbose=-1, seed=42,
)


def train_and_tune(cands, labels, true_count, n_s1, model_path, params_path,
                   val_frac=0.2, max_train_neg=12_000_000, backend='?'):
    import lightgbm as lgb

    rng = np.random.default_rng(42)
    val_s1 = rng.random(n_s1) < val_frac
    s1_idx = cands['s1_idx'].to_numpy()
    c_idx = cands['c_idx'].to_numpy()
    is_val = val_s1[s1_idx]

    tr = np.flatnonzero(~is_val)
    pos, neg = tr[labels[tr]], tr[~labels[tr]]
    if len(neg) > max_train_neg:
        neg = rng.choice(neg, max_train_neg, replace=False)
    tr = np.sort(np.concatenate([pos, neg]))
    va = np.flatnonzero(is_val)
    es = va if len(va) <= 5_000_000 else np.sort(rng.choice(va, 5_000_000, replace=False))
    print(f"  Train pairs: {len(tr):,} ({len(pos):,} positive) | validation pairs: {len(va):,}")

    dtrain = lgb.Dataset(feature_rows(cands, tr), labels[tr].astype(np.float32),
                         feature_name=FEATURE_COLS, free_raw_data=True)
    dval = lgb.Dataset(feature_rows(cands, es), labels[es].astype(np.float32),
                       reference=dtrain)
    t = time.time()
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(100)])
    print(f"  Trained {booster.best_iteration} rounds in {time.time() - t:.0f}s")
    del dtrain, dval

    imp = sorted(zip(FEATURE_COLS, booster.feature_importance('gain')), key=lambda x: -x[1])
    print("  Top features (gain):", ', '.join(f for f, _ in imp[:12]))

    prob_va = predict(booster, cands, va)
    sel_params, f05 = tune_selection(s1_idx[va], c_idx[va], prob_va, labels[va],
                                     true_count, val_s1)

    booster.save_model(model_path, num_iteration=booster.best_iteration)
    with open(params_path, 'w') as f:
        json.dump({**sel_params, 'val_f05': f05, 'backend': backend,
                   'features': FEATURE_COLS}, f, indent=2)
    print(f"  Saved model -> {model_path}\n  Saved params -> {params_path}")
    return booster, sel_params, f05


def feature_rows(cands, rows, cols=FEATURE_COLS):
    """float32 matrix of `cols` for the given row positions (copies only those rows)."""
    return np.column_stack([cands[c].to_numpy()[rows] for c in cols]).astype(np.float32, copy=False)


def predict(booster, cands, rows=None, chunk=5_000_000, cols=FEATURE_COLS):
    rows = np.arange(len(cands)) if rows is None else rows
    out = np.empty(len(rows), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = booster.predict(feature_rows(cands, rows[s:s + chunk], cols))
    return out


# ── candidate filter (second blocking stage) ─────────────────────────────────

PF_PARAMS = dict(
    objective='binary', learning_rate=0.1, num_leaves=63, min_data_in_leaf=500,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=42,
)


def train_prefilter(pf, cols, s1_idx, labels, n_s1, true_count, target_recall, model_path,
                    val_frac=0.2, max_train_neg=10_000_000):
    """Train the cheap candidate filter on training S1 entities and pick the lowest
    score that still keeps `target_recall` of the true pairs blocking found for the
    validation S1 entities (same S1 split and seed as the final model).
    Returns (scores for every pair, threshold)."""
    import lightgbm as lgb

    rng = np.random.default_rng(42)
    val_s1 = rng.random(n_s1) < val_frac
    is_val = val_s1[s1_idx]
    tr = np.flatnonzero(~is_val)
    pos, neg = tr[labels[tr]], tr[~labels[tr]]
    if len(neg) > max_train_neg:
        neg = rng.choice(neg, max_train_neg, replace=False)
    tr = np.sort(np.concatenate([pos, neg]))
    va = np.flatnonzero(is_val)
    es = va if len(va) <= 3_000_000 else np.sort(rng.choice(va, 3_000_000, replace=False))

    t = time.time()
    dtrain = lgb.Dataset(feature_rows(pf, tr, cols), labels[tr].astype(np.float32),
                         feature_name=list(cols), free_raw_data=True)
    dval = lgb.Dataset(feature_rows(pf, es, cols), labels[es].astype(np.float32), reference=dtrain)
    booster = lgb.train(PF_PARAMS, dtrain, num_boost_round=400, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(100)])
    del dtrain, dval
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    print(f"  Filter model: {booster.best_iteration} rounds in {time.time() - t:.0f}s -> {model_path}")

    score = predict(booster, pf, cols=cols)
    pos_val = score[va][labels[va]]
    threshold = float(np.quantile(pos_val, 1 - target_recall)) if len(pos_val) else 0.0
    keep = score >= threshold
    total = max(int(true_count.sum()), 1)
    print(f"  Filter threshold {threshold:.5f} keeps {target_recall:.1%} of blocking's true pairs "
          f"(validation S1s)")
    print(f"  Pairs: {len(score):,} -> {int(keep.sum()):,} "
          f"({len(score) / n_s1:.1f} -> {keep.sum() / n_s1:.1f} per S1)")
    print(f"  RECALL AFTER FILTER = {labels[keep].sum() / total:.4f} "
          f"(blocking alone {labels.sum() / total:.4f})")
    return score, threshold
