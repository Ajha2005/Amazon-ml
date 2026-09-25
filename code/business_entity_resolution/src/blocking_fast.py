"""
Fast inverted-index blocking for 2M+ S1 / 10M+ S2/S3.

Strategy (no TF-IDF — too slow at this scale):
  1. Partition by country (always consistent per EDA)
  2. Build token inverted index on S2/S3 name tokens
  3. For each S1, look up its name tokens → candidate S2/S3 records
  4. Only use tokens with IDF above threshold (discriminative tokens)
  5. Postal code exact-match blocking as a second pass
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import math


def build_inverted_index(df, text_col):
    """Build {token: [entity_id, ...]} index."""
    index = defaultdict(list)
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        text = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue
        for token in set(text.split()):
            if len(token) >= 3:
                index[token].append(row['entity_id'])
        if i % 500000 == 0 and i > 0:
            print(f"    Indexed {i:,}/{total:,} S2/S3 records...")
    return index


def compute_idf(index, total_docs):
    """IDF per token."""
    return {t: math.log(total_docs / (1 + len(ids))) for t, ids in index.items()}


def inverted_index_blocking(s1_part, s2s3_part, text_col,
                             min_idf=2.0, max_candidates_per_token=500):
    """
    For each S1, find S2/S3 records sharing at least one rare name token.
    min_idf=2.0 means token appears in <exp(-2)*N ≈ 13% of docs.
    max_candidates_per_token: skip tokens that are too common (hotword cap).
    """
    total_docs = len(s2s3_part)
    print(f"    Building inverted index on {total_docs:,} S2/S3 records...")
    index = build_inverted_index(s2s3_part, text_col)
    idf = compute_idf(index, total_docs)

    # Filter: only keep tokens with high enough IDF and not too many matches
    rare_index = {
        t: ids for t, ids in index.items()
        if idf.get(t, 0) >= min_idf and len(ids) <= max_candidates_per_token
    }
    print(f"    Retained {len(rare_index):,}/{len(index):,} discriminative tokens")

    pairs = []
    total_s1 = len(s1_part)
    for i, (_, row) in enumerate(s1_part.iterrows()):
        if i % 200000 == 0 and i > 0:
            print(f"    Looked up {i:,}/{total_s1:,} S1 records...")
        text = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue
        cands = set()
        for token in set(text.split()):
            if token in rare_index:
                cands.update(rare_index[token])
        for cand_id in cands:
            pairs.append((row['entity_id'], cand_id))

    return pairs


def postal_blocking(s1_part, s2s3_part):
    """Exact postal code match within country partition."""
    s2s3_postal = defaultdict(list)
    for _, row in s2s3_part.iterrows():
        p = row.get('postal_code', '')
        if p and len(p) >= 5:
            s2s3_postal[p].append(row['entity_id'])

    pairs = []
    for _, row in s1_part.iterrows():
        p = row.get('postal_code', '')
        if p and len(p) >= 5:
            for cand_id in s2s3_postal.get(p, []):
                pairs.append((row['entity_id'], cand_id))
    return pairs


def get_candidates_fast(s1_norm, s2s3_norm, top_k=None):
    """
    Country-partitioned inverted-index blocking.
    top_k is unused (kept for API compatibility) — inverted index
    naturally returns only records sharing discriminative tokens.
    Returns DataFrame(s1_id, cand_id).
    """
    all_pairs = []
    countries = s1_norm['country'].dropna().unique()
    print(f"  Countries in S1: {list(countries)}")

    for country in countries:
        s1_part = s1_norm[s1_norm['country'] == country].reset_index(drop=True)
        s2s3_part = s2s3_norm[s2s3_norm['country'] == country].reset_index(drop=True)

        if len(s2s3_part) == 0:
            print(f"  [{country}] No S2/S3 — skipping")
            continue

        print(f"\n  [{country}] S1={len(s1_part):,}, S2/S3={len(s2s3_part):,}")

        # Pass 1: token inverted index on cleaned name
        print(f"  [{country}] Inverted index blocking on name tokens...")
        pairs = inverted_index_blocking(
            s1_part, s2s3_part,
            text_col='name_no_suffix',
            min_idf=2.0,
            max_candidates_per_token=500,
        )
        print(f"    -> {len(pairs):,} pairs from token blocking")
        all_pairs.extend(pairs)

        # Pass 2: postal code exact match
        print(f"  [{country}] Postal code blocking...")
        postal_pairs = postal_blocking(s1_part, s2s3_part)
        print(f"    -> {len(postal_pairs):,} pairs from postal blocking")
        all_pairs.extend(postal_pairs)

    result = pd.DataFrame(all_pairs, columns=['s1_id', 'cand_id']).drop_duplicates()
    print(f"\n  Total unique candidate pairs: {len(result):,}")
    return result
