import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from . import config, split, ground_truth, blocking, normalize, address_features
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_audit():
    val_ids = list(split.get_val_s1_ids())
    random.seed(42)
    sample_ids = set(random.sample(val_ids, min(50000, len(val_ids))))
    
    truth_map = ground_truth.load_ground_truth()
    
    logger.info("Loading S1 Sample...")
    s1_df = pd.read_csv(config.TRAIN_SOURCE1, sep='\t', dtype=str).fillna("")
    s1_df = s1_df[s1_df['entity_id'].isin(sample_ids)]
    s1_df = normalize.normalize_dataframe(s1_df)
    s1_df = address_features.extract_address_features(s1_df)
    
    logger.info("Loading S2 and S3 for analysis...")
    s2_df = pd.read_csv(config.TRAIN_SOURCE2, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country']).fillna("")
    s3_df = pd.read_csv(config.TRAIN_SOURCE3, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country']).fillna("")
    s_cands = pd.concat([s2_df, s3_df], ignore_index=True)
    s_cands = normalize.normalize_dataframe(s_cands)
    s_cands = address_features.extract_address_features(s_cands)
    cand_map = s_cands.set_index('entity_id').to_dict('index')
    
    blocker = blocking.Blocker()
    blocker.load_indexes()
    
    cand_counts = []
    
    rule_stats = {
        'name_idx': {'cands': 0, 'unique_cands': 0, 'true_matches': 0, 'unique_true_matches': 0},
        'name_no_suffix_idx': {'cands': 0, 'unique_cands': 0, 'true_matches': 0, 'unique_true_matches': 0},
        'postal_idx': {'cands': 0, 'unique_cands': 0, 'true_matches': 0, 'unique_true_matches': 0},
        'name_token_idx': {'cands': 0, 'unique_cands': 0, 'true_matches': 0, 'unique_true_matches': 0},
        'address_token_idx': {'cands': 0, 'unique_cands': 0, 'true_matches': 0, 'unique_true_matches': 0},
    }
    
    total_true_to_find = 0
    total_true_found = 0
    
    missed_categories = {
        'missing_name': 0,
        'missing_address': 0,
        'different_country': 0,
        'postal_issue': 0,
        'other': 0
    }
    
    logger.info("Evaluating blocking rules...")
    for _, row in tqdm(s1_df.iterrows(), total=len(s1_df)):
        eid = row['entity_id']
        country = row['country']
        name_clean = row['business_name_clean']
        name_no_suf = row['business_name_without_legal_suffix']
        postal = row['postal_code']
        house = row['house_number']
        addr_clean = row['business_address_clean']
        
        t_matches = truth_map.get(eid, set())
        total_true_to_find += len(t_matches)
        
        r_cands = {r: set() for r in rule_stats.keys()}
        
        if name_clean: blocker._add_candidates(r_cands['name_idx'], "name_idx", f"{country}||{name_clean}")
        if name_no_suf: blocker._add_candidates(r_cands['name_no_suffix_idx'], "name_no_suffix_idx", f"{country}||{name_no_suf}")
        if postal: blocker._add_candidates(r_cands['postal_idx'], "postal_idx", f"{country}||{postal}", cap=5000)
        
        if name_no_suf:
            for token in blocking.get_tokens(name_no_suf):
                blocker._add_candidates(r_cands['name_token_idx'], "name_token_idx", f"{country}||{token}", cap=30000)
                
        if house and addr_clean:
            for token in blocking.get_tokens(addr_clean):
                if token != house:
                    blocker._add_candidates(r_cands['address_token_idx'], "address_token_idx", f"{country}||{house}||{token}", cap=3000)
                    
        all_cands = set()
        for r, c in r_cands.items():
            all_cands.update(c)
            
        cand_counts.append(len(all_cands))
        
        found_matches = all_cands.intersection(t_matches)
        total_true_found += len(found_matches)
        
        for r, c in r_cands.items():
            rule_stats[r]['cands'] += len(c)
            other_cands = set()
            for r2, c2 in r_cands.items():
                if r2 != r: other_cands.update(c2)
            
            rule_stats[r]['unique_cands'] += len(c - other_cands)
            r_true = c.intersection(t_matches)
            rule_stats[r]['true_matches'] += len(r_true)
            rule_stats[r]['unique_true_matches'] += len(r_true - other_cands)
            
        missed = t_matches - all_cands
        for m in missed:
            if m not in cand_map: continue
            cand = cand_map[m]
            
            if not name_clean or not cand['business_name_clean']:
                missed_categories['missing_name'] += 1
            elif not addr_clean or not cand['business_address_clean']:
                missed_categories['missing_address'] += 1
            elif country != cand['country']:
                missed_categories['different_country'] += 1
            elif postal and cand['postal_code'] and postal != cand['postal_code']:
                missed_categories['postal_issue'] += 1
            else:
                missed_categories['other'] += 1
                
    
    logger.info("=========================================")
    logger.info("         BLOCKING AUDIT REPORT           ")
    logger.info("=========================================")
    logger.info(f"Sample Size: {len(s1_df):,} queries")
    logger.info(f"Total True to Find: {total_true_to_find:,}")
    logger.info(f"Total True Found: {total_true_found:,}")
    logger.info(f"Recall: {total_true_found / max(1, total_true_to_find):.4f}")
    
    logger.info("\n--- CANDIDATE DISTRIBUTION ---")
    logger.info(f"Total Cands for Sample: {np.sum(cand_counts):,}")
    logger.info(f"Mean: {np.mean(cand_counts):.2f}")
    logger.info(f"Median: {np.median(cand_counts):.2f}")
    logger.info(f"P90: {np.percentile(cand_counts, 90):.2f}")
    logger.info(f"P95: {np.percentile(cand_counts, 95):.2f}")
    logger.info(f"P99: {np.percentile(cand_counts, 99):.2f}")
    logger.info(f"Max: {np.max(cand_counts):,}")
    
    logger.info("\n--- RULE STATS ---")
    for r, st in rule_stats.items():
        logger.info(f"RULE: {r}")
        logger.info(f"  Cands: {st['cands']:,} (Uniq: {st['unique_cands']:,})")
        logger.info(f"  True Matches: {st['true_matches']:,} (Uniq: {st['unique_true_matches']:,})")
        
    logger.info("\n--- MISSED CATEGORIES ---")
    for k, v in missed_categories.items():
        logger.info(f"  {k}: {v:,}")
    logger.info("=========================================")

if __name__ == "__main__":
    run_audit()
