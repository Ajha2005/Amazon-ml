"""
LightGBM scorer trained on ground truth pairs.
Falls back to gradient boosting from sklearn if lgb unavailable.
"""
import os
import pandas as pd
import numpy as np
import joblib

from features import FEATURE_COLS


def _get_model_class():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier, 'lgbm'
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        return XGBClassifier, 'xgb'
    except ImportError:
        pass
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier, 'sklearn'


def label_pairs(features_df, gt_df):
    """Add 'label' column: 1 = true match, 0 = non-match."""
    true_pairs = set()
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if matched and str(matched).strip():
            for cid in str(matched).split(','):
                cid = cid.strip()
                if cid:
                    true_pairs.add((s1_id, cid))

    labels = features_df.apply(
        lambda r: 1 if (r['s1_id'], r['cand_id']) in true_pairs else 0, axis=1
    )
    features_df = features_df.copy()
    features_df['label'] = labels
    pos = labels.sum()
    neg = (labels == 0).sum()
    print(f"  Labels: {pos:,} positives, {neg:,} negatives ({neg/max(pos,1):.1f}:1)")
    return features_df, true_pairs


def train_model(features_df, gt_df, model_path):
    """Train classifier on labeled candidate pairs."""
    features_df, _ = label_pairs(features_df, gt_df)

    X = features_df[FEATURE_COLS].fillna(0)
    y = features_df['label']

    pos = int(y.sum())
    neg = int((y == 0).sum())
    scale = neg / max(pos, 1)

    Cls, kind = _get_model_class()
    print(f"  Training {kind} on {len(X):,} pairs (scale_pos_weight={scale:.1f})...")

    if kind == 'lgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1,
        )
    elif kind == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='logloss',
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42,
        )

    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"  Model saved → {model_path}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp = sorted(zip(FEATURE_COLS, model.feature_importances_),
                     key=lambda x: -x[1])
        print("  Top features:", [(f, f'{v:.3f}') for f, v in imp[:8]])

    return model


def predict_scores(features_df, model):
    """Return features_df with 'score' column (match probability)."""
    X = features_df[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    out = features_df[['s1_id', 'cand_id']].copy()
    out['score'] = probs
    # Add key features so apply_threshold can still inspect them
    for col in ['name_jw', 'name_tsr', 'addr_tsr', 'postal_exact', 'name_exact']:
        if col in features_df.columns:
            out[col.replace('_tsr', '_ts')] = features_df[col].values
    out['name_ts']    = features_df.get('name_tsr', pd.Series(0, index=features_df.index)).values
    out['addr_ts']    = features_df.get('addr_tsr', pd.Series(0, index=features_df.index)).values
    out['postal_feat'] = features_df.get('postal_exact', pd.Series(0, index=features_df.index)).values
    out['exact']      = features_df.get('name_exact', pd.Series(0, index=features_df.index)).values
    return out


def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    print(f"  Loaded ML model from {model_path}")
    return model
