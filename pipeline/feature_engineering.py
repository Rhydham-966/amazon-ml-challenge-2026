import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from rapidfuzz.distance import JaroWinkler, Levenshtein
from . import config, normalize, address_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def jaccard_sim(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def token_overlap(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    return len(set1.intersection(set2))

def build_features(pairs_df, s1_data, s2_data):
    """
    Computes all pairwise features efficiently using Pandas and RapidFuzz (C++).
    Expects pairs_df to have ['s1_id', 's2_id', 'label' (optional)]
    """
    logger.info("Joining text data to pairs...")
    
    # Only keep needed columns in dataframes to save RAM
    cols_to_keep = ['entity_id', 'business_name_clean', 'business_name_without_legal_suffix', 
                    'business_address_clean', 'country', 'postal_code', 'house_number']
    
    s1_sub = s1_data[cols_to_keep]
    s2_sub = s2_data[cols_to_keep]
    
    # Left join S1 data
    df = pairs_df.merge(s1_sub, left_on='s1_id', right_on='entity_id', how='left')
    df = df.rename(columns={
        'business_name_clean': 'name1',
        'business_name_without_legal_suffix': 'name1_no_suf',
        'business_address_clean': 'addr1',
        'country': 'country1',
        'postal_code': 'postal1',
        'house_number': 'house1'
    })
    
    # Left join S2 data
    df = df.merge(s2_sub, left_on='s2_id', right_on='entity_id', how='left')
    df = df.rename(columns={
        'business_name_clean': 'name2',
        'business_name_without_legal_suffix': 'name2_no_suf',
        'business_address_clean': 'addr2',
        'country': 'country2',
        'postal_code': 'postal2',
        'house_number': 'house2'
    })
    
    # Clean NaNs
    text_cols = ['name1', 'name1_no_suf', 'addr1', 'country1', 'postal1', 'house1',
                 'name2', 'name2_no_suf', 'addr2', 'country2', 'postal2', 'house2']
    for c in text_cols:
        df[c] = df[c].fillna("")
        
    logger.info("Computing Exact Match & Boolean Features...")
    df['exact_name_match'] = (df['name1'] == df['name2']).astype(np.int8)
    df['exact_addr_match'] = ((df['addr1'] == df['addr2']) & (df['addr1'] != "")).astype(np.int8)
    df['postal_match'] = ((df['postal1'] == df['postal2']) & (df['postal1'] != "")).astype(np.int8)
    df['house_match'] = ((df['house1'] == df['house2']) & (df['house1'] != "")).astype(np.int8)
    df['country_match'] = (df['country1'] == df['country2']).astype(np.int8)
    
    df['name_missing'] = ((df['name1'] == "") | (df['name2'] == "")).astype(np.int8)
    df['addr_missing'] = ((df['addr1'] == "") | (df['addr2'] == "")).astype(np.int8)
    
    logger.info("Computing Length & Token Features...")
    df['name_len_diff'] = np.abs(df['name1'].str.len() - df['name2'].str.len())
    df['addr_len_diff'] = np.abs(df['addr1'].str.len() - df['addr2'].str.len())
    
    df['name1_tokens'] = df['name1'].apply(lambda x: len(x.split()))
    df['name2_tokens'] = df['name2'].apply(lambda x: len(x.split()))
    df['name_token_diff'] = np.abs(df['name1_tokens'] - df['name2_tokens'])
    
    logger.info("Computing Jaccard and Token Overlaps...")
    name_jaccard = []
    name_overlap = []
    addr_jaccard = []
    addr_overlap = []
    
    # Vectorized loop for custom python functions
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Token Metrics"):
        name_jaccard.append(jaccard_sim(row.name1, row.name2))
        name_overlap.append(token_overlap(row.name1, row.name2))
        addr_jaccard.append(jaccard_sim(row.addr1, row.addr2))
        addr_overlap.append(token_overlap(row.addr1, row.addr2))
        
    df['name_jaccard'] = np.array(name_jaccard, dtype=np.float32)
    df['name_token_overlap'] = np.array(name_overlap, dtype=np.int8)
    df['addr_jaccard'] = np.array(addr_jaccard, dtype=np.float32)
    df['addr_token_overlap'] = np.array(addr_overlap, dtype=np.int8)
    
    logger.info("Computing RapidFuzz String Distances (C++ Optimized)...")
    # Using list comprehensions with RapidFuzz is incredibly fast because it skips Pandas overhead
    df['name_jaro_winkler'] = np.array([JaroWinkler.normalized_similarity(a, b) for a, b in zip(df['name1'], df['name2'])], dtype=np.float32)
    df['name_no_suf_jaro'] = np.array([JaroWinkler.normalized_similarity(a, b) for a, b in zip(df['name1_no_suf'], df['name2_no_suf'])], dtype=np.float32)
    df['name_levenshtein'] = np.array([Levenshtein.normalized_similarity(a, b) for a, b in zip(df['name1'], df['name2'])], dtype=np.float32)
    df['addr_jaro_winkler'] = np.array([JaroWinkler.normalized_similarity(a, b) for a, b in zip(df['addr1'], df['addr2'])], dtype=np.float32)
    
    # Drop raw text columns to save memory before feeding to LightGBM
    drop_cols = text_cols + ['entity_id_x', 'entity_id_y', 'name1_tokens', 'name2_tokens']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    return df

def process_training_data():
    logger.info("Loading pairs dataset...")
    pairs_df = pd.read_parquet(config.DATA_DIR / "train_pairs.parquet")
    
    logger.info("Loading S1 data...")
    s1_df = pd.read_csv(config.TRAIN_SOURCE1, sep='\t', dtype=str).fillna("")
    s1_df = normalize.normalize_dataframe(s1_df)
    s1_df = address_features.extract_address_features(s1_df)
    
    logger.info("Loading S2 and S3 data...")
    s2_df = pd.read_csv(config.TRAIN_SOURCE2, sep='\t', dtype=str).fillna("")
    s3_df = pd.read_csv(config.TRAIN_SOURCE3, sep='\t', dtype=str).fillna("")
    s_cands = pd.concat([s2_df, s3_df], ignore_index=True)
    s_cands = normalize.normalize_dataframe(s_cands)
    s_cands = address_features.extract_address_features(s_cands)
    
    features_df = build_features(pairs_df, s1_df, s_cands)
    
    out_path = config.DATA_DIR / "train_features.parquet"
    logger.info(f"Saving training features to {out_path} ...")
    features_df.to_parquet(out_path, index=False)
    logger.info("Phase 9 Complete! ML Matrix is ready.")

if __name__ == "__main__":
    process_training_data()
