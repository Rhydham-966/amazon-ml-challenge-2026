import logging
from tqdm import tqdm
from . import config
from . import io_utils
from . import normalize
from . import address_features
from . import split
from . import ground_truth
from . import blocking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_blocking():
    logger.info("Initializing Blocking Evaluation...")
    val_ids = split.get_val_s1_ids()
    truth_map = ground_truth.load_ground_truth()
    
    blocker = blocking.Blocker()
    blocker.load_indexes()
    
    logger.info("Generating candidates for Source 1 Validation entities...")
    
    total_true_matches = 0
    found_true_matches = 0
    total_candidates = 0
    s1_processed = 0
    
    for chunk in tqdm(io_utils.read_tsv_chunked(config.TRAIN_SOURCE1), desc="Evaluating Blocking"):
        val_chunk = chunk[chunk['entity_id'].isin(val_ids)].copy()
        
        if val_chunk.empty:
            continue
            
        df = normalize.normalize_dataframe(val_chunk)
        df = address_features.extract_address_features(df)
        
        for _, row in df.iterrows():
            s1_id = row['entity_id']
            country = row['country']
            name_clean = row['business_name_clean']
            name_no_suf = row['business_name_without_legal_suffix']
            postal = row['postal_code']
            house = row['house_number']
            addr_clean = row['business_address_clean']
            
            candidates = blocker.get_candidates(country, name_clean, name_no_suf, postal, house, addr_clean)
            true_matches = ground_truth.get_true_matches(truth_map, s1_id)
            
            total_true_matches += len(true_matches)
            found_true_matches += len(true_matches.intersection(candidates))
            total_candidates += len(candidates)
            s1_processed += 1
            
    recall = found_true_matches / max(1, total_true_matches)
    avg_cands = total_candidates / max(1, s1_processed)
    
    logger.info(f"--- Advanced Blocking V2 Evaluation Results ---")
    logger.info(f"Validation Entities Processed: {s1_processed:,}")
    logger.info(f"Total True Matches to Find: {total_true_matches:,}")
    logger.info(f"True Matches Actually Found: {found_true_matches:,}")
    logger.info(f"Total Candidates Generated: {total_candidates:,}")
    logger.info(f"Average Candidates per Entity: {avg_cands:.2f}")
    logger.info(f"BLOCKING RECALL: {recall:.4f} ({recall*100:.2f}%)")

if __name__ == "__main__":
    evaluate_blocking()
