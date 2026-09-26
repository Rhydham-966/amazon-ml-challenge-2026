import pandas as pd
import numpy as np
import logging
import random
from pathlib import Path
from tqdm import tqdm
from . import config, split, ground_truth, blocking, normalize, address_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def build_pairs():
    logger.info("Initializing Phase 8: ML Pair Dataset Generation...")
    
    # Only use TRAIN data. No validation leak.
    train_ids = set(split.get_train_s1_ids())
    truth_map = ground_truth.load_ground_truth()
    
    # We sample 200,000 S1 train entities. 
    # Processing the entire 1.76M would take over an hour and generate 30M+ pairs, 
    # but LightGBM can achieve peak accuracy on 2M highly curated pairs in just minutes.
    random.seed(42)
    sample_ids = set(random.sample(list(train_ids), min(200000, len(train_ids))))
    
    logger.info("Loading S1 Train Sample...")
    s1_df = pd.read_csv(config.TRAIN_SOURCE1, sep='\t', dtype=str).fillna("")
    s1_df = s1_df[s1_df['entity_id'].isin(sample_ids)]
    s1_df = normalize.normalize_dataframe(s1_df)
    s1_df = address_features.extract_address_features(s1_df)
    
    blocker = blocking.Blocker()
    blocker.load_indexes()
    
    pairs = []
    
    for _, row in tqdm(s1_df.iterrows(), total=len(s1_df), desc="Sampling Pairs"):
        eid = row['entity_id']
        country = row['country']
        name_clean = row['business_name_clean']
        name_no_suf = row['business_name_without_legal_suffix']
        postal = row['postal_code']
        house = row['house_number']
        addr_clean = row['business_address_clean']
        
        t_matches = truth_map.get(eid, set())
        
        # We track how many rules generated each candidate to find HARD negatives
        cand_counts = {}
        
        def add_cand(c_set):
            for c in c_set:
                cand_counts[c] = cand_counts.get(c, 0) + 1
                
        # Rule 1 & 2
        r = set(); blocker._add_candidates(r, "name_idx", f"{country}||{name_clean}"); add_cand(r)
        r = set(); blocker._add_candidates(r, "name_no_suffix_idx", f"{country}||{name_no_suf}"); add_cand(r)
        
        # Rule 3
        if postal: 
            r = set(); blocker._add_candidates(r, "postal_idx", f"{country}||{postal}", cap=2000); add_cand(r)
            
        # Rule 4
        if name_no_suf:
            for token in blocking.get_tokens(name_no_suf):
                r = set(); blocker._add_candidates(r, "name_token_idx", f"{country}||{token}", cap=10000); add_cand(r)
                
        # Rule 5
        if house and addr_clean:
            for token in blocking.get_tokens(addr_clean):
                if token != house:
                    r = set(); blocker._add_candidates(r, "address_token_idx", f"{country}||{house}||{token}", cap=3000); add_cand(r)
                    
        all_cands = set(cand_counts.keys())
        found_positives = all_cands.intersection(t_matches)
        negatives = list(all_cands - t_matches)
        
        # Add Positives
        for p in found_positives:
            pairs.append({'s1_id': eid, 's2_id': p, 'label': 1})
            
        if not negatives:
            continue
            
        # Select Hard Negatives: sort candidates by how many rules they triggered
        # A candidate triggering 3 rules is much "harder" (more similar) than a candidate triggering 1 rule
        negatives.sort(key=lambda x: cand_counts[x], reverse=True)
        
        # Take up to 10 hard negatives
        hard_negs = negatives[:10]
        for n in hard_negs:
            pairs.append({'s1_id': eid, 's2_id': n, 'label': 0})
            
        # Take 1 random easy negative from the remainder
        remaining = negatives[10:]
        if remaining:
            easy = random.choice(remaining)
            pairs.append({'s1_id': eid, 's2_id': easy, 'label': 0})
            
    df_pairs = pd.DataFrame(pairs)
    logger.info("=========================================")
    logger.info("      TRAINING DATASET GENERATED         ")
    logger.info("=========================================")
    logger.info(f"Total Pairs: {len(df_pairs):,}")
    logger.info(f"Positives (Label 1): {len(df_pairs[df_pairs['label'] == 1]):,}")
    logger.info(f"Negatives (Label 0): {len(df_pairs[df_pairs['label'] == 0]):,}")
    
    out_path = config.DATA_DIR / "train_pairs.parquet"
    df_pairs.to_parquet(out_path, index=False)
    logger.info(f"Saved dataset successfully to: {out_path}")

if __name__ == "__main__":
    build_pairs()
