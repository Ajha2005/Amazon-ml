"""
Feature extraction for candidate pairs.
17 features covering name, address, postal, acronym similarity.
"""
import pandas as pd
import numpy as np
from difflib import SequenceMatcher


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
    return (matches/l1 + matches/l2 + (matches-t/2)/matches) / 3


def _jw(s1, s2, p=0.1):
    j = _jaro(s1, s2)
    px = sum(1 for i in range(min(4, len(s1), len(s2)))
             if s1[i] == s2[i] and all(s1[x] == s2[x] for x in range(i)))
    return j + px * p * (1 - j)


def _ratio(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()


def _tsr(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    inter, d1, d2 = t1&t2, t1-t2, t2-t1
    t0 = ' '.join(sorted(inter))
    a = ' '.join(sorted(inter)+sorted(d1))
    b = ' '.join(sorted(inter)+sorted(d2))
    return max(SequenceMatcher(None,t0,a).ratio(),
               SequenceMatcher(None,t0,b).ratio(),
               SequenceMatcher(None,a,b).ratio())


def _jaccard(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)


FEATURE_COLS = [
    'name_jw', 'name_tsr', 'name_ratio', 'name_exact',
    'name_first_word', 'name_last_word', 'name_jaccard',
    'name_len_ratio', 'name_common_tokens', 'name_len_diff',
    'suffix_match', 'acronym_match',
    'addr_tsr', 'addr_jw',
    'postal_exact', 'postal_prefix', 'postal_present',
]


def extract_features_batch(candidates_df, s1_norm, s2s3_norm, batch_size=10000):
    """
    Extract features for all candidate pairs.
    Returns DataFrame with s1_id, cand_id, label(if present), + FEATURE_COLS.
    """
    s1_dict = s1_norm.set_index('entity_id').to_dict('index')
    c_dict  = s2s3_norm.set_index('entity_id').to_dict('index')

    rows = []
    total = len(candidates_df)
    print(f"  Extracting features for {total:,} pairs...")

    for i, (_, row) in enumerate(candidates_df.iterrows()):
        if i % 500000 == 0 and i > 0:
            print(f"    {i:,}/{total:,}...")

        s1 = s1_dict.get(row['s1_id'], {})
        c  = c_dict.get(row['cand_id'], {})

        n1 = s1.get('name_no_suffix', '') or ''
        n2 = c.get('name_no_suffix', '') or ''
        a1 = s1.get('addr_expanded', '') or ''
        a2 = c.get('addr_expanded', '') or ''
        p1 = s1.get('postal_code', '') or ''
        p2 = c.get('postal_code', '') or ''
        ac1 = s1.get('acronym', '') or ''
        ac2 = c.get('acronym', '') or ''
        sf1 = s1.get('legal_suffix', '') or ''
        sf2 = c.get('legal_suffix', '') or ''

        t1 = n1.split()
        t2 = n2.split()
        common = len(set(t1) & set(t2))

        feat = {
            's1_id':   row['s1_id'],
            'cand_id': row['cand_id'],
            # Name similarity
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
            # Entity type
            'suffix_match':      float(sf1 == sf2 and sf1 != ''),
            'acronym_match':     float(
                (ac1 and ac1 == n2) or (ac2 and ac2 == n1) or
                (ac1 and ac2 and ac1 == ac2)
            ),
            # Address
            'addr_tsr':      _tsr(a1, a2),
            'addr_jw':       _jw(a1, a2),
            # Postal
            'postal_exact':   float(bool(p1 and p2 and p1 == p2)),
            'postal_prefix':  float(bool(p1 and p2 and len(p1)>=3 and len(p2)>=3 and p1[:3]==p2[:3])),
            'postal_present': float(bool(p1 and p2)),
        }
        rows.append(feat)

    return pd.DataFrame(rows)
