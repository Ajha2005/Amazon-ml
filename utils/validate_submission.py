#!/usr/bin/env python3
"""
Validates matching_results.tsv and candidate_pairs.tsv against the rules.
Usage:
    python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""
import argparse
import csv
import os
import sys


def load_tsv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            rows.append(row)
    return header, rows


def load_entity_ids(test_dir):
    s1_ids = set()
    s2_ids = set()
    s3_ids = set()
    for fname in ['test_source1.tsv']:
        with open(os.path.join(test_dir, fname)) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                s1_ids.add(row[0])
    for fname in ['test_source2.tsv']:
        with open(os.path.join(test_dir, fname)) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                s2_ids.add(row[0])
    for fname in ['test_source3.tsv']:
        with open(os.path.join(test_dir, fname)) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                s3_ids.add(row[0])
    return s1_ids, s2_ids, s3_ids


def validate(matching_path, candidate_path, test_dir):
    issues = []
    s1_ids, s2_ids, s3_ids = load_entity_ids(test_dir)
    valid_cand_ids = s2_ids | s3_ids

    # Validate matching_results.tsv
    header, rows = load_tsv(matching_path)
    if header != ['source1_entity_id', 'matched_entity_ids']:
        issues.append(f"matching_results.tsv: bad header {header}")

    seen_s1 = set()
    for row in rows:
        s1_id = row[0]
        if s1_id in seen_s1:
            issues.append(f"matching_results.tsv: duplicate S1 {s1_id}")
        seen_s1.add(s1_id)

        if len(row) > 1 and row[1].strip():
            match_ids = row[1].split(',')
            seen_match = set()
            for mid in match_ids:
                mid = mid.strip()
                if mid in seen_match:
                    issues.append(f"matching_results.tsv: duplicate match {mid} for {s1_id}")
                seen_match.add(mid)
                if mid not in valid_cand_ids:
                    issues.append(f"matching_results.tsv: {mid} not in test S2/S3 for {s1_id}")

    missing = s1_ids - seen_s1
    if missing:
        issues.append(f"matching_results.tsv: {len(missing)} S1 entities missing")

    extra = seen_s1 - s1_ids
    if extra:
        issues.append(f"matching_results.tsv: {len(extra)} unknown S1 entities")

    # Validate candidate_pairs.tsv
    header2, rows2 = load_tsv(candidate_path)
    if header2 != ['source1_entity_id', 'candidate_entity_ids']:
        issues.append(f"candidate_pairs.tsv: bad header {header2}")

    cand_dict = {}
    seen_s1_c = set()
    for row in rows2:
        s1_id = row[0]
        if s1_id in seen_s1_c:
            issues.append(f"candidate_pairs.tsv: duplicate S1 {s1_id}")
        seen_s1_c.add(s1_id)
        cands = set()
        if len(row) > 1 and row[1].strip():
            for cid in row[1].split(','):
                cid = cid.strip()
                if cid in cands:
                    issues.append(f"candidate_pairs.tsv: duplicate {cid} for {s1_id}")
                cands.add(cid)
                if cid not in valid_cand_ids:
                    issues.append(f"candidate_pairs.tsv: {cid} not in test S2/S3")
        cand_dict[s1_id] = cands

    missing_c = s1_ids - seen_s1_c
    if missing_c:
        issues.append(f"candidate_pairs.tsv: {len(missing_c)} S1 entities missing")

    # Check matches are subset of candidates
    for row in rows:
        s1_id = row[0]
        if len(row) > 1 and row[1].strip():
            match_ids = set(m.strip() for m in row[1].split(','))
            cands = cand_dict.get(s1_id, set())
            not_in_cands = match_ids - cands
            if not_in_cands:
                issues.append(f"WARNING: {s1_id} has matches not in candidates: {not_in_cands}")

    if issues:
        print("FAIL")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return 1
    else:
        print("PASS")
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matching', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--test-dir', required=True)
    args = parser.parse_args()
    sys.exit(validate(args.matching, args.candidate, args.test_dir))
