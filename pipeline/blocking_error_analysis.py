import pandas as pd
import logging
from . import config
from . import split
from . import ground_truth
from . import blocking
from . import normalize
from . import address_features
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_blocking_errors():
    val_ids = set(split.get_val_s1_ids())
    truth_map = ground_truth.load_ground_truth()
    
    # Load Source 1
    logger.info("Loading Validation Source 1...")
    s1_df = pd.read_csv(config.TRAIN_SOURCE1, sep='\t', dtype=str).fillna("")
    s1_df = s1_df[s1_df['entity_id'].isin(val_ids)]
    s1_df = normalize.normalize_dataframe(s1_df)
    s1_df = address_features.extract_address_features(s1_df)
    s1_map = s1_df.set_index('entity_id').to_dict('index')
    
    # We need Source 2 and Source 3 true matches to see what they look like
    logger.info("Loading Source 2 and 3 for ground truth text...")
    s2_df = pd.read_csv(config.TRAIN_SOURCE2, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country']).fillna("")
    s3_df = pd.read_csv(config.TRAIN_SOURCE3, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country']).fillna("")
    
    s_cands = pd.concat([s2_df, s3_df], ignore_index=True)
    s_cands = normalize.normalize_dataframe(s_cands)
    s_cands = address_features.extract_address_features(s_cands)
    cand_map = s_cands.set_index('entity_id').to_dict('index')
    
    blocker = blocking.Blocker()
    blocker.load_indexes()
    
    missed_pairs = []
    
    logger.info("Finding missed pairs...")
    for s1_id, row in s1_map.items():
        true_matches = truth_map.get(s1_id, set())
        if not true_matches:
            continue
            
        cands = blocker.get_candidates(
            row['country'], row['business_name_clean'], 
            row['business_name_without_legal_suffix'], 
            row['postal_code'], row['house_number'], 
            row['business_address_clean']
        )
        
        missed = true_matches - cands
        for m_id in missed:
            if m_id in cand_map:
                missed_pairs.append((row, cand_map[m_id]))
                if len(missed_pairs) >= 20:  # Just need a few examples
                    break
        if len(missed_pairs) >= 20:
            break
            
    logger.info("\n=== 20 EXAMPLES OF TRUE MATCHES WE MISSED ===")
    for i, (s1, s2) in enumerate(missed_pairs):
        logger.info(f"\n--- Missed Pair {i+1} ---")
        logger.info(f"S1: [Country: {s1['country']}] [Name: {s1['business_name']}] [Addr: {s1['business_address']}]")
        logger.info(f"    [Tokens: {blocking.get_tokens(s1['business_name_without_legal_suffix'])}] [Postal: {s1['postal_code']}] [House: {s1['house_number']}]")
        logger.info(f"S2: [Country: {s2['country']}] [Name: {s2['business_name']}] [Addr: {s2['business_address']}]")
        logger.info(f"    [Tokens: {blocking.get_tokens(s2['business_name_without_legal_suffix'])}] [Postal: {s2['postal_code']}] [House: {s2['house_number']}]")

if __name__ == "__main__":
    analyze_blocking_errors()
