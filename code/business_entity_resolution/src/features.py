"""
Feature extraction for candidate pairs — parallel version using fork.
Workers inherit lookup dicts via OS copy-on-write (Linux/Kaggle).
"""
import multiprocessing as mp
import pandas as pd
import numpy as np
from difflib import SequenceMatcher


# ── string similarity ──────────────────────────────────────────────────────────

def _jaro(s1, s2):
    if s1 == s2: return 1.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2: return 0.0
    md = max(l1, l2) // 2 - 1
    m1, m2 = [False]*l1, [False]*l2
    matches = 0
    for i in range(l1):
        for j in range(max(0, i-md), min(i+md+1, l2)):
            if not m2[j] and s1[i] == s2[j]:
                m1[i] = m2[j] = True; matches += 1; break
    if not matches: return 0.0
    t, k = 0, 0
    for i in range(l1):
        if not m1[i]: continue
        while not m2[k]: k += 1
        if s1[i] != s2[k]: t += 1
        k += 1
    return (matches/l1 + matches/l2 + (matches - t/2)/matches) / 3


def _jw(s1, s2, p=0.1):
    j = _jaro(s1, s2)
    px = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]: px += 1
        else: break
    return j + px * p * (1 - j)


def _ratio(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()


def _tsr(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    inter, d1, d2 = t1 & t2, t1 - t2, t2 - t1
    t0 = ' '.join(sorted(inter))
    a  = ' '.join(sorted(inter) + sorted(d1))
    b  = ' '.join(sorted(inter) + sorted(d2))
    return max(SequenceMatcher(None, t0, a).ratio(),
               SequenceMatcher(None, t0, b).ratio(),
               SequenceMatcher(None, a,  b).ratio())


def _jaccard(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    if not t1 and not t2: return 1.0
    if not t1 or  not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)


# ── feature column list ────────────────────────────────────────────────────────

FEATURE_COLS = [
    'name_jw', 'name_tsr', 'name_ratio', 'name_exact',
    'name_first_word', 'name_last_word', 'name_jaccard',
    'name_len_ratio', 'name_common_tokens', 'name_len_diff',
    'suffix_match', 'acronym_match',
    'addr_tsr', 'addr_jw',
    'postal_exact', 'postal_prefix', 'postal_present',
]


# ── global lookup dicts (inherited by fork workers) ────────────────────────────

_W_S1: dict = {}
_W_C:  dict = {}


def _build_lookup(norm_df: pd.DataFrame) -> dict:
    """
    Build compact tuple lookup: entity_id -> (name_no_suffix, addr_expanded,
    postal_code, acronym, legal_suffix).
    Vectorized — fast even for 10M rows.
    """
    cols = ['name_no_suffix', 'addr_expanded', 'postal_code', 'acronym', 'legal_suffix']
    present = [c for c in cols if c in norm_df.columns]
    sub = norm_df[['entity_id'] + present].fillna('')
    # pad missing columns with empty strings
    for c in cols:
        if c not in sub.columns:
            sub[c] = ''
    arrays = [sub[c].values for c in cols]
    return {eid: tuple(a[i] for a in arrays)
            for i, eid in enumerate(sub['entity_id'].values)}


# ── worker function (module-level so fork can inherit it) ─────────────────────

def _worker_extract(pair_list: list) -> list:
    """Extract features for one chunk of (s1_id, cand_id) pairs."""
    rows = []
    for s1_id, cand_id in pair_list:
        sv = _W_S1.get(s1_id, ('', '', '', '', ''))
        cv = _W_C.get(cand_id,  ('', '', '', '', ''))
        n1, a1, p1, ac1, sf1 = sv
        n2, a2, p2, ac2, sf2 = cv

        t1 = n1.split()
        t2 = n2.split()
        common = len(set(t1) & set(t2))

        rows.append({
            's1_id':   s1_id,
            'cand_id': cand_id,
            'name_jw':           _jw(n1, n2),
            'name_tsr':          _tsr(n1, n2),
            'name_ratio':        _ratio(n1, n2),
            'name_exact':        float(n1 == n2 and n1 != ''),
            'name_first_word':   float(bool(t1 and t2 and t1[0] == t2[0])),
            'name_last_word':    float(bool(t1 and t2 and t1[-1] == t2[-1])),
            'name_jaccard':      _jaccard(n1, n2),
            'name_len_ratio':    min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
            'name_common_tokens': common,
            'name_len_diff':     abs(len(t1) - len(t2)),
            'suffix_match':      float(sf1 == sf2 and sf1 != ''),
            'acronym_match':     float(
                bool((ac1 and ac1 == n2) or (ac2 and ac2 == n1) or
                     (ac1 and ac2 and ac1 == ac2))
            ),
            'addr_tsr':      _tsr(a1, a2),
            'addr_jw':       _jw(a1, a2),
            'postal_exact':  float(bool(p1 and p2 and p1 == p2)),
            'postal_prefix': float(bool(p1 and p2 and len(p1) >= 3
                                        and len(p2) >= 3 and p1[:3] == p2[:3])),
            'postal_present': float(bool(p1 and p2)),
        })
    return rows


# ── public API ─────────────────────────────────────────────────────────────────

def extract_features_batch(candidates_df, s1_norm, s2s3_norm,
                            n_workers=None, batch_size=None):
    """
    Extract features for all candidate pairs.
    Uses multiprocessing fork on Linux (Kaggle) for ~3-4x speedup.
    n_workers=None → auto (min(4, cpu_count)); set to 1 to disable parallelism.
    """
    global _W_S1, _W_C

    total = len(candidates_df)

    # Determine workers
    if n_workers is None:
        n_workers = min(4, mp.cpu_count())
    print(f"  Building compact lookups for {len(s1_norm):,} S1 + {len(s2s3_norm):,} S2/S3...")
    _W_S1 = _build_lookup(s1_norm)
    _W_C  = _build_lookup(s2s3_norm)

    pairs = list(zip(candidates_df['s1_id'].values,
                     candidates_df['cand_id'].values))

    if n_workers == 1:
        print(f"  Extracting {total:,} pairs (single-threaded)...")
        all_rows = _worker_extract(pairs)
    else:
        chunk_size = (total + n_workers - 1) // n_workers
        chunks = [pairs[i:i + chunk_size] for i in range(0, total, chunk_size)]
        print(f"  Extracting {total:,} pairs with {len(chunks)} workers "
              f"({chunk_size:,} pairs each)...")
        ctx = mp.get_context('fork')   # fork inherits globals — Linux only
        with ctx.Pool(len(chunks)) as pool:
            results = pool.map(_worker_extract, chunks)
        all_rows = [row for chunk in results for row in chunk]

    print(f"  Feature extraction done — {len(all_rows):,} rows")
    return pd.DataFrame(all_rows)
