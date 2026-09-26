"""
Pair features, computed column-wise over all candidate pairs at once.

String similarities use rapidfuzz's C++ pairwise scorer (process.cpdist) when it
is installed and fall back to difflib in forked worker processes otherwise.
Everything else is numpy on integer codes, so 50M pairs fit comfortably in RAM.
"""
import multiprocessing as mp
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    BACKEND = 'rapidfuzz'
except ImportError:
    BACKEND = 'difflib'

STRING_FEATURES = [
    ('name_ratio', 'name_no_suffix', 'ratio'),
    ('name_tsr', 'name_no_suffix', 'token_set'),
    ('name_tsort', 'name_no_suffix', 'token_sort'),
    ('name_partial', 'name_no_suffix', 'partial'),
    ('name_jw', 'name_no_suffix', 'jw'),
    ('fullname_tsr', 'name_clean', 'token_set'),
    ('addr_ratio', 'addr_expanded', 'ratio'),
    ('addr_tsr', 'addr_expanded', 'token_set'),
    ('addr_partial', 'addr_expanded', 'partial'),
    ('addr_jw', 'addr_expanded', 'jw'),
]

FEATURE_COLS = [f for f, _, _ in STRING_FEATURES] + [
    'name_exact', 'first_tok_eq', 'last_tok_eq',
    'name_len1', 'name_len2', 'name_len_ratio', 'ntok1', 'ntok2',
    'acr_s1_eq_name', 'acr_c_eq_name', 'acr_eq',
    'suffix_eq', 'suffix_both',
    'postal_eq', 'postal_both', 'house_eq', 'house_both', 'addr_both',
    'src',
    'block_rel', 'block_rank', 'n_cands_s1', 'n_s1_c',
    'base_rank_s1', 'base_gap_s1', 'base_rank_c', 'base_gap_c',
]


# ── difflib fallback ──────────────────────────────────────────────────────────

def _jw_py(s1, s2):
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2:
        return 0.0
    md = max(max(l1, l2) // 2 - 1, 0)
    m1, m2 = [False] * l1, [False] * l2
    m = 0
    for i in range(l1):
        for j in range(max(0, i - md), min(i + md + 1, l2)):
            if not m2[j] and s1[i] == s2[j]:
                m1[i] = m2[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t = k = 0
    for i in range(l1):
        if m1[i]:
            while not m2[k]:
                k += 1
            t += s1[i] != s2[k]
            k += 1
    j = (m / l1 + m / l2 + (m - t / 2) / m) / 3
    p = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a != b:
            break
        p += 1
    return j + p * 0.1 * (1 - j)


def _ratio_py(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _token_sort_py(a, b):
    return _ratio_py(' '.join(sorted(a.split())), ' '.join(sorted(b.split())))


def _token_set_py(a, b):
    t1, t2 = set(a.split()), set(b.split())
    inter = ' '.join(sorted(t1 & t2))
    x = (inter + ' ' + ' '.join(sorted(t1 - t2))).strip()
    y = (inter + ' ' + ' '.join(sorted(t2 - t1))).strip()
    if inter and (not (t1 - t2) or not (t2 - t1)):
        return 1.0
    return max(_ratio_py(inter, x), _ratio_py(inter, y), _ratio_py(x, y))


def _partial_py(a, b):
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    best = 0.0
    for blk in SequenceMatcher(None, short, long_).get_matching_blocks():
        start = max(blk[1] - blk[0], 0)
        r = _ratio_py(short, long_[start:start + len(short)])
        if r > best:
            best = r
            if r > 0.995:
                break
    return best


_PY_FUNCS = {'ratio': _ratio_py, 'token_set': _token_set_py,
             'token_sort': _token_sort_py, 'partial': _partial_py, 'jw': _jw_py}
_W = {}


def _py_job(args):
    kind, start, end = args
    f, a, b = _PY_FUNCS[kind], _W['a'], _W['b']
    return np.fromiter((f(a[i], b[i]) for i in range(start, end)),
                       dtype=np.float32, count=end - start)


def _pairwise(a, b, kind, workers, chunk=2_000_000):
    n = len(a)
    if BACKEND == 'rapidfuzz':
        scorer = {'ratio': fuzz.ratio, 'token_set': fuzz.token_set_ratio,
                  'token_sort': fuzz.token_sort_ratio, 'partial': fuzz.partial_ratio,
                  'jw': JaroWinkler.normalized_similarity}[kind]
        scale = 1.0 if kind == 'jw' else 0.01
        out = np.empty(n, dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            out[s:e] = process.cpdist(a[s:e], b[s:e], scorer=scorer, workers=workers) * scale
        return out
    _W['a'], _W['b'] = a, b
    step = 200_000
    jobs = [(kind, s, min(s + step, n)) for s in range(0, n, step)]
    with mp.get_context('fork').Pool(workers) as pool:
        parts = pool.map(_py_job, jobs, chunksize=1)
    _W.clear()
    return np.concatenate(parts) if parts else np.empty(0, np.float32)


# ── record-level attributes ───────────────────────────────────────────────────

def _joint_codes(*series):
    """Factorize several string columns together so equal strings share a code;
    empty strings get code -1."""
    arrays = [s.fillna('').to_numpy(dtype=object) for s in series]
    codes, uniques = pd.factorize(np.concatenate(arrays))
    empty = np.flatnonzero(uniques == '')
    if len(empty):
        codes[codes == empty[0]] = -1
    out, pos = [], 0
    for a in arrays:
        out.append(codes[pos:pos + len(a)])
        pos += len(a)
    return out


def _eq(a, b):
    return ((a == b) & (a >= 0)).astype(np.float32)


def _group_rank_gap(key, value):
    s = pd.Series(value)
    g = s.groupby(key)
    rank = g.rank(ascending=False, method='min').to_numpy(dtype=np.float32)
    gap = (g.transform('max') - s).to_numpy(dtype=np.float32)
    return rank, gap


def compute_features(cands, s1_norm, c_norm, workers=None):
    """Adds FEATURE_COLS to `cands` (DataFrame with s1_idx, c_idx, block_score, block_rank)."""
    workers = workers or min(4, mp.cpu_count())
    i1 = cands['s1_idx'].to_numpy()
    i2 = cands['c_idx'].to_numpy()
    n = len(cands)
    print(f"  Computing features for {n:,} pairs (string backend: {BACKEND}, {workers} workers)")

    t = time.time()
    for feat, col, kind in STRING_FEATURES:
        a = s1_norm[col].fillna('').to_numpy(dtype=object)[i1]
        b = c_norm[col].fillna('').to_numpy(dtype=object)[i2]
        cands[feat] = _pairwise(a, b, kind, workers)
        print(f"    {feat:<14} {time.time() - t:6.0f}s")
    del a, b

    n1, n2 = s1_norm['name_no_suffix'].fillna(''), c_norm['name_no_suffix'].fillna('')
    name1, name2, acr1, acr2 = _joint_codes(n1, n2, s1_norm['acronym'], c_norm['acronym'])
    cands['name_exact'] = _eq(name1[i1], name2[i2])
    cands['acr_s1_eq_name'] = _eq(acr1[i1], name2[i2])
    cands['acr_c_eq_name'] = _eq(acr2[i2], name1[i1])
    cands['acr_eq'] = _eq(acr1[i1], acr2[i2])

    f1, f2 = _joint_codes(n1.str.split(' ', n=1).str[0], n2.str.split(' ', n=1).str[0])
    cands['first_tok_eq'] = _eq(f1[i1], f2[i2])
    l1, l2 = _joint_codes(n1.str.rsplit(' ', n=1).str[-1], n2.str.rsplit(' ', n=1).str[-1])
    cands['last_tok_eq'] = _eq(l1[i1], l2[i2])

    len1 = n1.str.len().to_numpy(dtype=np.float32)[i1]
    len2 = n2.str.len().to_numpy(dtype=np.float32)[i2]
    cands['name_len1'], cands['name_len2'] = len1, len2
    cands['name_len_ratio'] = np.minimum(len1, len2) / np.maximum(np.maximum(len1, len2), 1)
    cands['ntok1'] = (n1.str.count(' ').to_numpy(dtype=np.float32) + (n1 != '').to_numpy())[i1]
    cands['ntok2'] = (n2.str.count(' ').to_numpy(dtype=np.float32) + (n2 != '').to_numpy())[i2]

    for name, col in [('suffix', 'legal_suffix'), ('postal', 'postal_code'), ('house', 'house_number')]:
        c1, c2 = _joint_codes(s1_norm[col], c_norm[col])
        a, b = c1[i1], c2[i2]
        cands[f'{name}_eq'] = _eq(a, b)
        cands[f'{name}_both'] = ((a >= 0) & (b >= 0)).astype(np.float32)
    a1 = (s1_norm['addr_expanded'].fillna('') != '').to_numpy()
    a2 = (c_norm['addr_expanded'].fillna('') != '').to_numpy()
    cands['addr_both'] = (a1[i1] & a2[i2]).astype(np.float32)

    cands['src'] = c_norm['src'].to_numpy(dtype=np.float32)[i2]

    # Context: how this pair compares with the other candidates of the same S1
    # record and with the other S1 records competing for the same S2/S3 record.
    bs = cands['block_score'].to_numpy()
    cands['block_rel'] = (bs / pd.Series(bs).groupby(i1).transform('max').to_numpy()).astype(np.float32)
    cands['block_rank'] = cands['block_rank'].astype(np.float32)
    cands['n_cands_s1'] = np.bincount(i1)[i1].astype(np.float32)
    cands['n_s1_c'] = np.bincount(i2)[i2].astype(np.float32)
    base = ((cands['name_tsr'] + cands['name_jw'] + cands['addr_tsr']) / 3).to_numpy()
    cands['base_rank_s1'], cands['base_gap_s1'] = _group_rank_gap(i1, base)
    cands['base_rank_c'], cands['base_gap_c'] = _group_rank_gap(i2, base)

    print(f"  Features done in {time.time() - t:.0f}s")
    return cands
