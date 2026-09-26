import pandas as pd
import numpy as np
import logging
from . import config, split, ground_truth

logging.basicConfig(level=logging.INFO, format='%(message)s')

def audit():
    df = pd.read_parquet(config.DATA_DIR / "train_pairs.parquet")
    gt = ground_truth.load_ground_truth()
    val_ids = split.get_val_s1_ids()
    train_ids = split.get_train_s1_ids()
    
    total = len(df)
    df['label'] = df['label'].astype(int)
    pos = len(df[df['label'] == 1])
    neg = len(df[df['label'] == 0])
    
    print(f"Total pairs: {total}")
    print(f"Positives: {pos}")
    print(f"Negatives: {neg}")
    print(f"Positive:Negative Ratio: 1 : {neg/max(1,pos):.2f}")
    
    unique_s1 = df['s1_id'].nunique()
    print(f"Unique S1 entities: {unique_s1}")
    
    val_overlap = set(df['s1_id']).intersection(set(val_ids))
    print(f"Validation S1 entities in training dataset: {len(val_overlap)}")
    
    # Ground truth verification
    valid_pos = 0
    invalid_pos = 0
    for _, r in df[df['label'] == 1].iterrows():
        if r['s2_id'] in gt.get(r['s1_id'], set()):
            valid_pos += 1
        else:
            invalid_pos += 1
            
    valid_neg = 0
    invalid_neg = 0
    for _, r in df[df['label'] == 0].iterrows():
        if r['s2_id'] not in gt.get(r['s1_id'], set()):
            valid_neg += 1
        else:
            invalid_neg += 1
            
    print(f"Positives in Ground Truth: {valid_pos} (Invalid: {invalid_pos})")
    print(f"Negatives NOT in Ground Truth: {valid_neg} (Invalid: {invalid_neg})")
    
    # Positive distribution
    pos_counts = df[df['label'] == 1].groupby('s1_id').size()
    print(f"Positive Pairs per S1 - Mean: {pos_counts.mean():.2f}, Max: {pos_counts.max()}")
    
    # Hard vs Random ratio
    neg_counts = df[df['label'] == 0].groupby('s1_id').size()
    est_hard = 0
    est_easy = 0
    for c in neg_counts:
        if c <= 10:
            est_hard += c
        else:
            est_hard += 10
            est_easy += (c - 10)
            
    print(f"Estimated Hard Negatives: {est_hard} ({est_hard/neg*100:.1f}%)")
    print(f"Estimated Random Negatives: {est_easy} ({est_easy/neg*100:.1f}%)")

if __name__ == "__main__":
    audit()
