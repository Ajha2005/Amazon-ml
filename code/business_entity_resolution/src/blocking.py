import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def tfidf_blocking(s1_df, s2s3_df, text_col, top_k=20, analyzer='char_wb', ngram_range=(3, 4)):
    corpus = pd.concat([s1_df[text_col], s2s3_df[text_col]], ignore_index=True)
    corpus = corpus.fillna('')

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=50000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    n_s1 = len(s1_df)
    s1_vectors = tfidf_matrix[:n_s1]
    s2s3_vectors = tfidf_matrix[n_s1:]

    s1_ids = s1_df['entity_id'].values
    s2s3_ids = s2s3_df['entity_id'].values

    pairs = []
    batch_size = 500
    for start in range(0, n_s1, batch_size):
        end = min(start + batch_size, n_s1)
        sims = cosine_similarity(s1_vectors[start:end], s2s3_vectors)
        for i in range(sims.shape[0]):
            row_idx = start + i
            top_indices = np.argsort(sims[i])[::-1][:top_k]
            top_indices = top_indices[sims[i][top_indices] > 0.01]
            for j in top_indices:
                pairs.append((s1_ids[row_idx], s2s3_ids[j], float(sims[i][j])))

    return pd.DataFrame(pairs, columns=['s1_id', 'cand_id', f'sim_{text_col}'])


def shared_token_blocking(s1_df, s2s3_df, text_col, min_idf=2.0):
    from collections import Counter

    all_texts = pd.concat([s1_df[text_col], s2s3_df[text_col]]).fillna('')
    doc_freq = Counter()
    total_docs = len(all_texts)
    for text in all_texts:
        tokens = set(text.split())
        for t in tokens:
            doc_freq[t] += 1

    idf = {t: np.log(total_docs / (1 + f)) for t, f in doc_freq.items()}

    s2s3_token_index = defaultdict(list)
    for idx, row in s2s3_df.iterrows():
        text = row[text_col] if isinstance(row[text_col], str) else ''
        for token in set(text.split()):
            if idf.get(token, 0) >= min_idf:
                s2s3_token_index[token].append(row['entity_id'])

    pairs = []
    for _, row in s1_df.iterrows():
        text = row[text_col] if isinstance(row[text_col], str) else ''
        cands = set()
        for token in set(text.split()):
            if idf.get(token, 0) >= min_idf:
                cands.update(s2s3_token_index.get(token, []))
        for cand_id in cands:
            pairs.append((row['entity_id'], cand_id))

    return pd.DataFrame(pairs, columns=['s1_id', 'cand_id']).drop_duplicates()


def postal_name_blocking(s1_df, s2s3_df):
    s1_with_postal = s1_df[s1_df['postal_code'] != ''].copy()
    s2s3_with_postal = s2s3_df[s2s3_df['postal_code'] != ''].copy()

    if len(s1_with_postal) == 0 or len(s2s3_with_postal) == 0:
        return pd.DataFrame(columns=['s1_id', 'cand_id'])

    s2s3_postal_index = defaultdict(list)
    for _, row in s2s3_with_postal.iterrows():
        s2s3_postal_index[row['postal_code']].append(row['entity_id'])

    s1_name_tokens = {}
    for _, row in s1_with_postal.iterrows():
        name = row['name_no_suffix'] if isinstance(row['name_no_suffix'], str) else ''
        s1_name_tokens[row['entity_id']] = set(name.split())

    s2s3_name_tokens = {}
    for _, row in s2s3_with_postal.iterrows():
        name = row['name_no_suffix'] if isinstance(row['name_no_suffix'], str) else ''
        s2s3_name_tokens[row['entity_id']] = set(name.split())

    pairs = []
    for _, row in s1_with_postal.iterrows():
        postal = row['postal_code']
        s1_tokens = s1_name_tokens[row['entity_id']]
        for cand_id in s2s3_postal_index.get(postal, []):
            cand_tokens = s2s3_name_tokens.get(cand_id, set())
            if s1_tokens & cand_tokens:
                pairs.append((row['entity_id'], cand_id))

    return pd.DataFrame(pairs, columns=['s1_id', 'cand_id']).drop_duplicates()


def get_candidates(s1_df, s2s3_df, top_k=30):
    print("  TF-IDF blocking on name (char n-grams)...")
    name_pairs = tfidf_blocking(s1_df, s2s3_df, 'name_no_suffix', top_k=top_k,
                                analyzer='char_wb', ngram_range=(3, 4))

    print("  TF-IDF blocking on name+address (word)...")
    s1_df = s1_df.copy()
    s2s3_df = s2s3_df.copy()
    s1_df['name_addr'] = s1_df['name_no_suffix'].fillna('') + ' ' + s1_df['addr_expanded'].fillna('')
    s2s3_df['name_addr'] = s2s3_df['name_no_suffix'].fillna('') + ' ' + s2s3_df['addr_expanded'].fillna('')

    addr_pairs = tfidf_blocking(s1_df, s2s3_df, 'name_addr', top_k=top_k,
                                analyzer='word', ngram_range=(1, 2))

    print("  Shared rare token blocking on name...")
    token_pairs = shared_token_blocking(s1_df, s2s3_df, 'name_no_suffix', min_idf=2.0)

    print("  Postal+name blocking...")
    postal_pairs = postal_name_blocking(s1_df, s2s3_df)

    all_pairs = pd.concat([
        name_pairs[['s1_id', 'cand_id']],
        addr_pairs[['s1_id', 'cand_id']],
        token_pairs[['s1_id', 'cand_id']],
        postal_pairs[['s1_id', 'cand_id']],
    ], ignore_index=True).drop_duplicates()

    print(f"  Total unique candidate pairs: {len(all_pairs)}")
    return all_pairs
