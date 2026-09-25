import pandas as pd
import numpy as np
import sys
import os

def run_eda(data_dir='dataset/train'):
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    s1 = pd.read_csv(os.path.join(data_dir, 'train_source1.tsv'), sep='\t')
    s2 = pd.read_csv(os.path.join(data_dir, 'train_source2.tsv'), sep='\t')
    s3 = pd.read_csv(os.path.join(data_dir, 'train_source3.tsv'), sep='\t')
    gt = pd.read_csv(os.path.join(data_dir, 'train_ground_truth.tsv'), sep='\t')

    print(f"\nDataset sizes:")
    print(f"  Source 1: {len(s1)} records")
    print(f"  Source 2: {len(s2)} records")
    print(f"  Source 3: {len(s3)} records")
    print(f"  Ground truth: {len(gt)} rows")

    print(f"\nColumn names:")
    print(f"  S1: {list(s1.columns)}")
    print(f"  S2: {list(s2.columns)}")
    print(f"  S3: {list(s3.columns)}")
    print(f"  GT: {list(gt.columns)}")

    print(f"\nSample S1 records:")
    print(s1.head(3).to_string())
    print(f"\nSample S2 records:")
    print(s2.head(3).to_string())
    print(f"\nSample S3 records:")
    print(s3.head(3).to_string())

    # Q1: Singleton fraction
    gt['match_list'] = gt['matched_entity_ids'].apply(
        lambda x: [] if pd.isna(x) or str(x).strip() == '' else str(x).split(',')
    )
    gt['match_count'] = gt['match_list'].apply(len)
    singletons = (gt['match_count'] == 0).sum()
    total_s1 = len(gt)
    print(f"\n--- Q1: Singleton fraction ---")
    print(f"  Singletons (no matches): {singletons}/{total_s1} = {singletons/total_s1:.4f}")

    # Q2: Distribution of match counts
    print(f"\n--- Q2: Match count distribution ---")
    print(gt['match_count'].describe().to_string())
    print(f"\n  Value counts:")
    vc = gt['match_count'].value_counts().sort_index()
    for count, freq in vc.items():
        print(f"    {count} matches: {freq} entities ({freq/total_s1*100:.1f}%)")

    # Q3: S2 vs S3 noise comparison
    print(f"\n--- Q3: S2 vs S3 noise comparison ---")
    print(f"  S2 missing business_name: {s2['business_name'].isna().sum()}/{len(s2)}")
    print(f"  S3 missing business_name: {s3['business_name'].isna().sum()}/{len(s3)}")
    print(f"  S2 missing business_address: {s2['business_address'].isna().sum()}/{len(s2)}")
    print(f"  S3 missing business_address: {s3['business_address'].isna().sum()}/{len(s3)}")
    print(f"  S2 missing country: {s2['country'].isna().sum()}/{len(s2)}")
    print(f"  S3 missing country: {s3['country'].isna().sum()}/{len(s3)}")
    print(f"  S2 avg name length: {s2['business_name'].dropna().str.len().mean():.1f}")
    print(f"  S3 avg name length: {s3['business_name'].dropna().str.len().mean():.1f}")
    print(f"  S2 avg address length: {s2['business_address'].dropna().str.len().mean():.1f}")
    print(f"  S3 avg address length: {s3['business_address'].dropna().str.len().mean():.1f}")

    # Q4: Does any S2/S3 ID appear under multiple S1?
    print(f"\n--- Q4: Uniqueness of S2/S3 across S1 entities ---")
    all_matches = []
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        for cand_id in row['match_list']:
            all_matches.append((s1_id, cand_id.strip()))
    matches_df = pd.DataFrame(all_matches, columns=['s1_id', 'cand_id'])

    if len(matches_df) > 0:
        cand_s1_counts = matches_df.groupby('cand_id')['s1_id'].nunique()
        multi_s1 = (cand_s1_counts > 1).sum()
        print(f"  S2/S3 records matched to >1 S1: {multi_s1}/{len(cand_s1_counts)}")
        if multi_s1 == 0:
            print("  >> Each S2/S3 record belongs to at most one S1 -- HUGE precision lever!")
        else:
            print(f"  >> WARNING: {multi_s1} records appear under multiple S1 entities")
            print("  Top offenders:")
            print(cand_s1_counts[cand_s1_counts > 1].sort_values(ascending=False).head(5).to_string())

    # Q5: Country consistency
    print(f"\n--- Q5: Country consistency between matches ---")
    print(f"  S1 countries: {s1['country'].value_counts().to_dict()}")
    print(f"  S2 countries: {s2['country'].value_counts().to_dict()}")
    print(f"  S3 countries: {s3['country'].value_counts().to_dict()}")

    s1_country = dict(zip(s1['entity_id'], s1['country']))
    s2_country = dict(zip(s2['entity_id'], s2['country']))
    s3_country = dict(zip(s3['entity_id'], s3['country']))
    all_country = {**s2_country, **s3_country}

    consistent = 0
    inconsistent = 0
    for _, row in matches_df.iterrows():
        s1_c = s1_country.get(row['s1_id'], '')
        cand_c = all_country.get(row['cand_id'], '')
        if pd.isna(s1_c) or pd.isna(cand_c):
            continue
        if s1_c == cand_c:
            consistent += 1
        else:
            inconsistent += 1

    total_checked = consistent + inconsistent
    if total_checked > 0:
        print(f"  Country consistent: {consistent}/{total_checked} = {consistent/total_checked:.4f}")
        print(f"  Country inconsistent: {inconsistent}/{total_checked}")
        if inconsistent == 0:
            print("  >> Country is always consistent -- can block on it (but don't for France)")
        else:
            print("  >> Country sometimes inconsistent -- use as feature, not filter")

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'dataset/train'
    run_eda(data_dir)
