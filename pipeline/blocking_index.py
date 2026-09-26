import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
from . import io_utils
from . import config
from . import normalize
from . import address_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STOPWORDS = {'the', 'and', 'of', 'in', 'to', 'for', 'on', 'at', 'by', 'a', 'an'}

def get_tokens(text: str):
    if not text: return []
    # Extract tokens longer than 2 characters, skipping stopwords
    return [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]

def process_and_index(source_path: Path, source_name: str):
    logger.info(f"Building Advanced Inverted Index for {source_name}...")
    
    name_idx = {}
    name_no_suffix_idx = {}
    postal_idx = {}
    name_token_idx = {}
    address_token_idx = {}
    
    try:
        for chunk in tqdm(io_utils.read_tsv_chunked(source_path), desc=f"Indexing {source_name}"):
            df = normalize.normalize_dataframe(chunk)
            df = address_features.extract_address_features(df)
            
            for _, row in df.iterrows():
                eid = row['entity_id']
                country = row['country']
                
                name_clean = row['business_name_clean']
                name_no_suf = row['business_name_without_legal_suffix']
                postal = row['postal_code']
                house = row['house_number']
                addr_clean = row['business_address_clean']
                
                # Rule 1 & 2: Exact Names
                if name_clean:
                    name_idx.setdefault(f"{country}||{name_clean}", []).append(eid)
                if name_no_suf and name_no_suf != name_clean:
                    name_no_suffix_idx.setdefault(f"{country}||{name_no_suf}", []).append(eid)
                    
                # Rule 3: Postal Code
                if postal:
                    postal_idx.setdefault(f"{country}||{postal}", []).append(eid)
                    
                # Rule 4: Name Tokens (Highly effective for fuzzy recall)
                if name_no_suf:
                    for token in get_tokens(name_no_suf):
                        name_token_idx.setdefault(f"{country}||{token}", []).append(eid)
                        
                # Rule 5: House Number + Address Token (Great for missing names)
                if house and addr_clean:
                    for token in get_tokens(addr_clean):
                        if token != house:
                            address_token_idx.setdefault(f"{country}||{house}||{token}", []).append(eid)
                            
    except Exception as e:
        logger.error(f"Error during indexing {source_name}: {str(e)}")
        raise
        
    return name_idx, name_no_suffix_idx, postal_idx, name_token_idx, address_token_idx

def merge_indexes(idx1, idx2):
    merged = idx1.copy()
    for k, v in idx2.items():
        if k in merged:
            merged[k].extend(v)
        else:
            merged[k] = v
    return merged

def save_index(idx: dict, name: str):
    path = config.INDEX_DIR / f"{name}.parquet"
    logger.info(f"Converting {name} dictionary to Parquet...")
    df = pd.DataFrame({
        'blocking_key': list(idx.keys()),
        'candidate_ids': list(idx.values())
    })
    df.to_parquet(path, engine='pyarrow', index=False)
    logger.info(f"Saved {name} index: {len(df):,} unique keys.")

def build_all_indexes():
    paths = io_utils.get_source_paths(is_train=True)
    
    s2_res = process_and_index(paths['source2'], "Source 2")
    s3_res = process_and_index(paths['source3'], "Source 3")
    
    logger.info("Merging Source 2 and Source 3 indexes...")
    indexes_to_save = [
        ("name_idx", s2_res[0], s3_res[0]),
        ("name_no_suffix_idx", s2_res[1], s3_res[1]),
        ("postal_idx", s2_res[2], s3_res[2]),
        ("name_token_idx", s2_res[3], s3_res[3]),
        ("address_token_idx", s2_res[4], s3_res[4])
    ]
    
    for name, s2_idx, s3_idx in indexes_to_save:
        final_idx = merge_indexes(s2_idx, s3_idx)
        save_index(final_idx, name)
        # Free memory immediately
        del final_idx
    
    logger.info("Advanced Blocking Index Construction Complete.")

if __name__ == "__main__":
    build_all_indexes()
