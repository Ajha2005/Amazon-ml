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


def _char_ngrams(texts, n=3):
    """Character n-grams inside word boundaries, so typos that break token overlap
    still share features (e.g. 'kacom' and 'kakom' share 'kak' or 'kac' but words differ)."""
    out = []
    for s in texts:
        if not s:
            out.append('')
            continue
        grams = []
        for w in s.split():
            if len(w) < n:
                grams.append('#' + w)
            else:
                for i in range(len(w) - n + 1):
                    grams.append(w[i:i + n])
        out.append(' '.join(grams))
    return out


_VOWELS = set('aeiouyh')


def _phonetic_word(w):
    """Consonant-only skeleton, dedupe consecutive dups, keep first 5. Groups
    similar-sounding words: 'solutions'->'sltns', 'solutionz'->'sltnz' share sltn."""
    cons = [c for c in w if c.isalpha() and c not in _VOWELS]
    deduped = []
    for c in cons:
        if not deduped or deduped[-1] != c:
            deduped.append(c)
    key = ''.join(deduped)[:5]
    return key or w[:2]


def _phonetic(texts):
    return [' '.join(_phonetic_word(w) for w in s.split() if w) for s in texts]


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
        'name_char': _char_ngrams(name, 3),
        'name_phon': _phonetic(name),
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


def _topk_rows(query, index_t, k, chunk, workers):
    """Top-k columns of query @ index_t for every row of query."""
    _G['x1'], _G['bt'], _G['k'] = query, index_t, k
    n = query.shape[0]
    bounds = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    if workers > 1 and 'fork' in mp.get_all_start_methods():
        with mp.get_context('fork').Pool(workers) as pool:
            parts = pool.map(_topk_chunk, bounds, chunksize=1)
    else:
        parts = [_topk_chunk(b) for b in bounds]
    _G.clear()
    return [np.concatenate(p) for p in zip(*parts)]


def _block_partition(s1_part, c_part, k, kc, df_cap, chunk, workers):
    x1, x2 = _field_matrices(s1_part, c_part)
    n_c = x2.shape[0]
    df = np.asarray(x2.sum(axis=0)).ravel()
    keep_cols = np.flatnonzero((df > 0) & (df <= df_cap))
    idf = sp.diags(np.log(n_c / df[keep_cols]).astype(np.float32))
    print(f"    vocab {len(df):,} tokens, kept {len(keep_cols):,} with df <= {df_cap}")
    x1 = x1[:, keep_cols].tocsr()
    x2 = x2[:, keep_cols].tocsr()

    # S1 -> S2/S3: best k records for each S1 entity
    s_a, c_a, sc_a, rk_a = _topk_rows((x1 @ idf).tocsr(), x2.T.tocsr(), k, chunk, workers)
    # S2/S3 -> S1: each S2/S3 record belongs to at most one S1, so its true S1 is
    # almost always among its best few. This recovers matches of S1 entities whose
    # own top-k is crowded out by look-alikes (chains, generic names).
    c_b, s_b, sc_b, rk_b = _topk_rows((x2 @ idf).tocsr(), x1.T.tocsr(), kc, chunk, workers)
    del x1, x2

    ka = s_a.astype(np.int64) * n_c + c_a
    kb = s_b.astype(np.int64) * n_c + c_b
    keys, inv = np.unique(np.concatenate([ka, kb]), return_inverse=True)
    ia, ib = inv[:len(ka)], inv[len(ka):]
    score = np.zeros(len(keys), np.float32)
    score[ia], score[ib] = sc_a, sc_b
    rank_s1 = np.full(len(keys), k, np.int16)
    rank_s1[ia] = rk_a
    rank_c = np.full(len(keys), kc, np.int16)
    rank_c[ib] = rk_b
    print(f"    S1-side {len(ka):,} + S2/S3-side {len(kb):,} -> {len(keys):,} unique pairs")
    return ((keys // n_c).astype(np.int32), (keys % n_c).astype(np.int32),
            score, rank_s1, rank_c)


def get_candidates_topk(s1_norm, s2s3_norm, k=50, kc=5, df_cap=1000, chunk=4000, workers=None):
    """
    Returns DataFrame(s1_idx, c_idx, block_score, block_rank, block_rank_c): the
    union of each S1 record's top-k S2/S3 records and each S2/S3 record's top-kc
    S1 records. Indices are row positions in s1_norm / s2s3_norm; block_rank is the
    pair's rank among the S1's candidates (k if only found from the S2/S3 side),
    block_rank_c its rank among the S2/S3 record's candidates (kc if not found there).
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
        r, c, score, rank, rank_c = _block_partition(
            s1_norm.iloc[s1_pos], s2s3_norm.iloc[c_pos], k, kc, df_cap, chunk, workers)
        out.append(pd.DataFrame({
            's1_idx': s1_pos[r].astype(np.int32),
            'c_idx': c_pos[c].astype(np.int32),
            'block_score': score,
            'block_rank': rank,
            'block_rank_c': rank_c,
        }))
        print(f"    -> {len(r):,} pairs in {time.time() - t:.0f}s")

    cands = pd.concat(out, ignore_index=True)
    print(f"  Total candidate pairs: {len(cands):,} "
          f"({len(cands) / max(len(s1_norm), 1):.1f} per S1 record)")
    return cands
