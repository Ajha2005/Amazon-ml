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
    total = int(true_count.sum())
    rank = cands['block_rank'].to_numpy()
    print(f"  Ground-truth pairs: {total:,}")
    for k in (1, 3, 5, 10, 15, 20, 30, 50):
        if k > rank.max() + 1:
            break
        hit = int(labels[rank < k].sum())
        n = int((rank < k).sum())
        print(f"    recall@{k:<3} = {hit / max(total, 1):.4f}   ({n:,} pairs)")


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
    objective='binary', learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, seed=42,
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
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
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


def feature_rows(cands, rows):
    """float32 matrix of FEATURE_COLS for the given row positions (copies only those rows)."""
    return np.column_stack([cands[c].to_numpy()[rows] for c in FEATURE_COLS]).astype(np.float32, copy=False)


def predict(booster, cands, rows=None, chunk=5_000_000):
    rows = np.arange(len(cands)) if rows is None else rows
    out = np.empty(len(rows), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = booster.predict(feature_rows(cands, rows[s:s + chunk]))
    return out
