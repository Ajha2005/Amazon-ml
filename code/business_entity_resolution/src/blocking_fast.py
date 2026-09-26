"""
Candidate generation: IDF-weighted token overlap, top-K per Source 1 record.

Name, address and postal tokens are vectorized into one sparse vocabulary
(separate vocabularies per field, so "main" in a name and "main" in an address
are different tokens). Composite tokens — adjacent-word bigrams and
name-word@postcode — stay rare for chains and generic names whose single words
are too common to block on. Tokens present in more than `df_cap` Source 2/3
records are dropped, the rest are weighted by IDF, and S1 x S2/S3 overlap
scores come from a chunked sparse matrix product. Only the `k` best-scoring
candidates per S1 record are kept. Runs independently per country.
"""
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

_G = {}


def _text(df, col):
    return df[col].fillna('').to_numpy(dtype=object)


def _bigrams(texts):
    return [' '.join(a + '_' + b for a, b in zip(t, t[1:])) for t in (s.split() for s in texts)]


def _fields(df):
    name = _text(df, 'name_no_suffix')
    addr = _text(df, 'addr_expanded')
    postal = _text(df, 'postal_code')
    return {
        'name': name,
        'addr': addr,
        'postal': postal,
        'name_bigram': _bigrams(name),
        'addr_bigram': _bigrams(addr),
        'name_postal': [' '.join(w + '@' + p for w in s.split()) if p else ''
                        for s, p in zip(name, postal)],
    }


def _field_matrices(s1_part, c_part):
    f1, f2 = _fields(s1_part), _fields(c_part)
    x1, x2 = [], []
    for field in f2:
        vec = CountVectorizer(token_pattern=r'\S+', lowercase=False,
                              binary=True, dtype=np.float32)
        try:
            b = vec.fit_transform(f2[field])
        except ValueError:  # field empty everywhere in this partition
            continue
        x2.append(b)
        x1.append(vec.transform(f1[field]))
    return sp.hstack(x1).tocsr(), sp.hstack(x2).tocsr()


def _topk_chunk(bounds):
    start, end = bounds
    k = _G['k']
    s = (_G['x1'][start:end] @ _G['bt']).tocsr()
    counts = np.diff(s.indptr)
    if s.nnz == 0:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32), np.empty(0, np.int16))
    rows = np.repeat(np.arange(end - start, dtype=np.int64), counts)
    order = np.argsort(rows * 1e6 - s.data, kind='stable')
    rank = np.arange(s.nnz) - np.repeat(s.indptr[:-1], counts)
    keep = rank < k
    sel = order[keep]
    return ((rows[sel] + start).astype(np.int32), s.indices[sel].astype(np.int32),
            s.data[sel].astype(np.float32), rank[keep].astype(np.int16))


def _block_partition(s1_part, c_part, k, df_cap, chunk, workers):
    x1, x2 = _field_matrices(s1_part, c_part)
    n_c = x2.shape[0]
    df = np.asarray(x2.sum(axis=0)).ravel()
    keep_cols = np.flatnonzero((df > 0) & (df <= df_cap))
    idf = np.log(n_c / df[keep_cols]).astype(np.float32)
    print(f"    vocab {len(df):,} tokens, kept {len(keep_cols):,} with df <= {df_cap}")

    _G['x1'] = (x1[:, keep_cols] @ sp.diags(idf)).tocsr()
    _G['bt'] = x2[:, keep_cols].T.tocsr()
    _G['k'] = k
    del x1, x2

    n1 = s1_part.shape[0]
    bounds = [(i, min(i + chunk, n1)) for i in range(0, n1, chunk)]
    if workers > 1 and 'fork' in mp.get_all_start_methods():
        with mp.get_context('fork').Pool(workers) as pool:
            parts = pool.map(_topk_chunk, bounds, chunksize=1)
    else:
        parts = [_topk_chunk(b) for b in bounds]
    _G.clear()
    return [np.concatenate(p) for p in zip(*parts)]


def get_candidates_topk(s1_norm, s2s3_norm, k=50, df_cap=1000, chunk=4000, workers=None):
    """
    Returns DataFrame(s1_idx, c_idx, block_score, block_rank) where the indices
    are row positions in s1_norm / s2s3_norm and block_rank is 0 for the best
    candidate of each S1 record.
    """
    workers = workers or min(4, mp.cpu_count())
    s1_country = s1_norm['country'].fillna('').to_numpy(dtype=object)
    c_country = s2s3_norm['country'].fillna('').to_numpy(dtype=object)
    countries = pd.unique(s1_country)
    print(f"  Countries in S1: {list(countries)} | in S2/S3: {list(pd.unique(c_country))}")

    out = []
    for country in countries:
        s1_pos = np.flatnonzero(s1_country == country)
        c_pos = np.flatnonzero(c_country == country)
        if len(c_pos) == 0:
            print(f"  [{country}] no S2/S3 records — {len(s1_pos):,} S1 records get no candidates")
            continue
        t = time.time()
        print(f"  [{country}] S1={len(s1_pos):,}  S2/S3={len(c_pos):,}")
        r, c, score, rank = _block_partition(
            s1_norm.iloc[s1_pos], s2s3_norm.iloc[c_pos], k, df_cap, chunk, workers)
        out.append(pd.DataFrame({
            's1_idx': s1_pos[r].astype(np.int32),
            'c_idx': c_pos[c].astype(np.int32),
            'block_score': score,
            'block_rank': rank,
        }))
        print(f"    -> {len(r):,} pairs in {time.time() - t:.0f}s")

    cands = pd.concat(out, ignore_index=True)
    print(f"  Total candidate pairs: {len(cands):,} "
          f"({len(cands) / max(len(s1_norm), 1):.1f} per S1 record)")
    return cands
