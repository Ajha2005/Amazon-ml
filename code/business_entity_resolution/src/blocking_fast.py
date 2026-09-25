"""
Scalable blocking for 2M+ S1 records against 10M+ S2/S3 records.
Strategy:
  1. Partition by country (always consistent — confirmed by EDA)
  2. Within each country: token inverted index + batched TF-IDF sparse cosine
  3. Postal code exact match within country
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import scipy.sparse as sp


def build_token_index(df, text_col):
    """Inverted index: token -> list of entity_ids"""
    index = defaultdict(list)
    for _, row in df.iterrows():
        text = row[text_col]
        if not isinstance(text, str):
            continue
        for token in set(text.split()):
            if len(token) >= 3:
                index[token].append(row['entity_id'])
    return index


def token_idf(index, total_docs):
    """IDF for each token"""
    return {t: np.log(total_docs / (1 + len(ids))) for t, ids in index.items()}


def token_blocking_partition(s1_part, s2s3_part, text_col, min_idf=3.0):
    """Shared rare token blocking within a country partition."""
    total = len(s1_part) + len(s2s3_part)
    s2s3_index = build_token_index(s2s3_part, text_col)
    idf = token_idf(s2s3_index, total)

    pairs = []
    for _, row in s1_part.iterrows():
        text = row[text_col]
        if not isinstance(text, str):
            continue
        cands = set()
        for token in set(text.split()):
            if idf.get(token, 0) >= min_idf and len(token) >= 3:
                cands.update(s2s3_index.get(token, []))
        for cand_id in cands:
            pairs.append((row['entity_id'], cand_id))
    return pairs


def tfidf_blocking_batched(s1_part, s2s3_part, text_col, top_k=10,
                            batch_size=2000, ngram_range=(3, 4)):
    """
    Batched TF-IDF cosine blocking. Fits vectorizer on full corpus,
    then scores S1 against S2/S3 in batches to avoid OOM.
    """
    s1_texts = s1_part[text_col].fillna('').tolist()
    s2s3_texts = s2s3_part[text_col].fillna('').tolist()
    s1_ids = s1_part['entity_id'].tolist()
    s2s3_ids = s2s3_part['entity_id'].tolist()

    print(f"    Fitting TF-IDF on {len(s1_texts)+len(s2s3_texts)} docs...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        max_features=200000,
        sublinear_tf=True,
        min_df=2,
    )
    all_texts = s1_texts + s2s3_texts
    vectorizer.fit(all_texts)

    print(f"    Transforming S2/S3 ({len(s2s3_texts)} docs)...")
    s2s3_matrix = vectorizer.transform(s2s3_texts)

    pairs = []
    n_s1 = len(s1_ids)
    n_batches = (n_s1 + batch_size - 1) // batch_size
    print(f"    Scoring {n_s1} S1 in {n_batches} batches...")

    for b in range(n_batches):
        if b % 10 == 0:
            print(f"      Batch {b}/{n_batches}...")
        start = b * batch_size
        end = min(start + batch_size, n_s1)
        s1_batch_texts = s1_texts[start:end]
        s1_batch_ids = s1_ids[start:end]

        s1_matrix = vectorizer.transform(s1_batch_texts)
        sims = cosine_similarity(s1_matrix, s2s3_matrix)

        for i in range(sims.shape[0]):
            top_idx = np.argpartition(sims[i], -top_k)[-top_k:]
            top_idx = top_idx[sims[i][top_idx] > 0.05]
            for j in top_idx:
                pairs.append((s1_batch_ids[i], s2s3_ids[j]))

    return pairs


def postal_blocking_partition(s1_part, s2s3_part):
    """Exact postal code match within country."""
    s2s3_postal = defaultdict(list)
    for _, row in s2s3_part.iterrows():
        p = row.get('postal_code', '')
        if p:
            s2s3_postal[p].append(row['entity_id'])

    pairs = []
    for _, row in s1_part.iterrows():
        p = row.get('postal_code', '')
        if p:
            for cand_id in s2s3_postal.get(p, []):
                pairs.append((row['entity_id'], cand_id))
    return pairs


def get_candidates_fast(s1_norm, s2s3_norm, top_k=10):
    """
    Country-partitioned blocking pipeline.
    Returns DataFrame(s1_id, cand_id).
    """
    all_pairs = []
    countries = s1_norm['country'].unique()
    print(f"  Countries in S1: {list(countries)}")

    for country in countries:
        s1_part = s1_norm[s1_norm['country'] == country].reset_index(drop=True)
        s2s3_part = s2s3_norm[s2s3_norm['country'] == country].reset_index(drop=True)

        if len(s2s3_part) == 0:
            print(f"  [{country}] No S2/S3 records — skipping")
            continue

        print(f"\n  [{country}] S1={len(s1_part)}, S2/S3={len(s2s3_part)}")

        # 1. TF-IDF on name (char n-grams)
        print(f"  [{country}] TF-IDF char n-gram blocking on names...")
        pairs = tfidf_blocking_batched(
            s1_part, s2s3_part, 'name_no_suffix',
            top_k=top_k, batch_size=2000, ngram_range=(3, 4)
        )
        print(f"    -> {len(pairs)} pairs")
        all_pairs.extend(pairs)

        # 2. Token blocking on name
        print(f"  [{country}] Token blocking on names...")
        pairs = token_blocking_partition(s1_part, s2s3_part, 'name_no_suffix', min_idf=3.0)
        print(f"    -> {len(pairs)} pairs")
        all_pairs.extend(pairs)

        # 3. Postal code blocking
        print(f"  [{country}] Postal code blocking...")
        pairs = postal_blocking_partition(s1_part, s2s3_part)
        print(f"    -> {len(pairs)} pairs")
        all_pairs.extend(pairs)

    result = pd.DataFrame(all_pairs, columns=['s1_id', 'cand_id']).drop_duplicates()
    print(f"\n  Total unique candidate pairs: {len(result)}")
    return result
