import pandas as pd
import numpy as np
import logging
import argparse
import time
import os
import psutil
import gc
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb

from . import config, blocking, normalize, address_features
from .feature_engineering import build_features
from .blocking_index import process_and_index, merge_indexes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_submission(chunk_size: int, threshold: float, dry_run: int = 0):
    start_time = time.time()
    logger.info("Initializing Final Submission Pipeline")
    
    logger.info("Building indexes for TEST Source 2 and Source 3...")
    s2_res = process_and_index(config.TEST_SOURCE2, "Test Source 2")
    s3_res = process_and_index(config.TEST_SOURCE3, "Test Source 3")
    
    test_indexes = {
        "name_idx": merge_indexes(s2_res[0], s3_res[0]),
        "name_no_suffix_idx": merge_indexes(s2_res[1], s3_res[1]),
        "postal_idx": merge_indexes(s2_res[2], s3_res[2]),
        "name_token_idx": merge_indexes(s2_res[3], s3_res[3]),
        "address_token_idx": merge_indexes(s2_res[4], s3_res[4])
    }
    
    blocker = blocking.Blocker()
    blocker.indexes = test_indexes
    
    model_path = config.MODELS_DIR / "lgbm_model.txt"
    booster = lgb.Booster(model_file=str(model_path))
    logger.info(f"Loaded trained LightGBM model. Using threshold {threshold}")
    
    logger.info("Loading TEST Source 1...")
    s1_df = pd.read_csv(config.TEST_SOURCE1, sep='\t', dtype=str).fillna("")
    if dry_run > 0:
        logger.info(f"DRY RUN: limiting S1 to {dry_run} entities.")
        s1_df = s1_df.head(dry_run)
        
    s1_df = normalize.normalize_dataframe(s1_df)
    s1_df = address_features.extract_address_features(s1_df)
    
    logger.info("Loading TEST Source 2/3 Data...")
    s2_df = pd.read_csv(config.TEST_SOURCE2, sep='\t', dtype=str).fillna("")
    s3_df = pd.read_csv(config.TEST_SOURCE3, sep='\t', dtype=str).fillna("")
    s_cands = pd.concat([s2_df, s3_df], ignore_index=True)
    s_cands = normalize.normalize_dataframe(s_cands)
    s_cands = address_features.extract_address_features(s_cands)
    
    logger.info("Evaluating candidates in chunks...")
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_pairs_file = config.OUTPUT_DIR / "candidate_pairs.tsv"
    matching_results_file = config.OUTPUT_DIR / "matching_results.tsv"
    
    f_cand = open(candidate_pairs_file, 'w', encoding='utf-8')
    f_match = open(matching_results_file, 'w', encoding='utf-8')
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    
    s1_batch_size = 5000
    
    global_cand_counts = []
    total_matches = 0
    total_candidates = 0
    total_features_time = 0
    
    def process_chunk(c_list, batch_results):
        nonlocal total_features_time
        if not c_list: return
        
        t0 = time.time()
        df_chunk = pd.DataFrame(c_list)
        features_chunk = build_features(df_chunk, s1_df, s_cands)
        drop_cols = ['s1_id', 's2_id', 'label']
        X = features_chunk.drop(columns=drop_cols, errors='ignore')
        probs = booster.predict(X)
        t1 = time.time()
        total_features_time += (t1 - t0)
        
        for pair, prob in zip(c_list, probs):
            s1_id = pair['s1_id']
            s2_id = pair['s2_id']
            batch_results[s1_id]['cands'].append(s2_id)
            if prob >= threshold:
                batch_results[s1_id]['matches'].append(s2_id)
                
    peak_mem = get_memory_usage()
    
    # Process S1 entities in bounded batches
    for i in tqdm(range(0, len(s1_df), s1_batch_size), desc="S1 Batches"):
        s1_batch = s1_df.iloc[i:i+s1_batch_size]
        # Initialize results for exactly this batch
        batch_results = {row.entity_id: {'cands': [], 'matches': []} for row in s1_batch.itertuples()}
        chunk_list = []
        
        for row in s1_batch.itertuples():
            cands = blocker.get_candidates(
                row.country, row.business_name_clean, 
                row.business_name_without_legal_suffix, 
                row.postal_code, row.house_number, 
                row.business_address_clean
            )
            
            for c in cands:
                chunk_list.append({'s1_id': row.entity_id, 's2_id': c})
                if len(chunk_list) >= chunk_size:
                    process_chunk(chunk_list, batch_results)
                    chunk_list = []
                    peak_mem = max(peak_mem, get_memory_usage())
                    
        # Process any remaining items in the chunk for this batch
        if chunk_list:
            process_chunk(chunk_list, batch_results)
            peak_mem = max(peak_mem, get_memory_usage())
            
        # Immediately write this batch to disk
        for s1_id in batch_results:
            cands_set = set(batch_results[s1_id]['cands'])
            match_set = set(batch_results[s1_id]['matches'])
            
            cands_str = ",".join(sorted(cands_set))
            match_str = ",".join(sorted(match_set))
            
            f_cand.write(f"{s1_id}\t{cands_str}\n")
            f_match.write(f"{s1_id}\t{match_str}\n")
            
            # Stats tracking
            cand_len = len(cands_set)
            global_cand_counts.append(cand_len)
            total_candidates += cand_len
            total_matches += len(match_set)
            
        # Release batch data from RAM
        del batch_results
        gc.collect()
        
    f_cand.close()
    f_match.close()
    
    elapsed = time.time() - start_time
    
    # Stats Reporting
    logger.info("\n=============================================")
    logger.info("           INFERENCE STATS                   ")
    logger.info("=============================================")
    logger.info(f"Number of test S1 entities: {len(s1_df):,}")
    logger.info(f"Total candidates generated: {total_candidates:,}")
    if global_cand_counts:
        logger.info(f"Average candidates/S1:      {np.mean(global_cand_counts):.1f}")
        logger.info(f"Median candidates/S1:       {np.median(global_cand_counts):.1f}")
        logger.info(f"P95 candidates/S1:          {np.percentile(global_cand_counts, 95):.1f}")
    logger.info(f"Total predicted matches:    {total_matches:,}")
    logger.info(f"Peak RAM usage:             {peak_mem:.2f} MB")
    logger.info(f"Total Runtime:              {elapsed:.1f}s")
    
    if elapsed > 0:
        cands_sec = total_candidates / elapsed
        features_sec = total_candidates / total_features_time if total_features_time > 0 else 0
        logger.info(f"Candidates/sec (overall):   {cands_sec:.1f}")
        logger.info(f"Features/sec (feature gen): {features_sec:.1f}")
    
    import os
    size_cand = os.path.getsize(candidate_pairs_file) / (1024*1024)
    size_match = os.path.getsize(matching_results_file) / (1024*1024)
    logger.info(f"candidate_pairs.tsv size:   {size_cand:.2f} MB")
    logger.info(f"matching_results.tsv size:  {size_match:.2f} MB")
    logger.info("=============================================\n")
    
    if dry_run > 0:
        logger.info("DRY RUN COMPLETE.")
        # Estimate full runtime assuming 2.2M test S1 entities (or exact if known, we'll estimate from known sample speed)
        # Using exact 175MB test file size to guess: 2.2M is roughly the size
        estimated_total_s1 = 2200000 
        estimated_runtime = (elapsed / len(s1_df)) * estimated_total_s1
        logger.info(f"Estimated full test runtime ({estimated_total_s1:,} S1): {estimated_runtime/3600:.2f} hours")
        return
        
    logger.info("Validating format...")
    import sys
    sys.path.append(str(config.ROOT_DIR))
    from utils.validate_submission import validate_submission
    errors = []
    validate_submission(
        str(matching_results_file),
        str(candidate_pairs_file),
        str(config.TEST_DIR),
        errors
    )
    if errors:
        logger.error("Validation failed! Errors:")
        for e in errors:
            logger.error(e)
    else:
        logger.info("Validation Result: PASSED! Ready to submit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--dry-run", type=int, default=0, help="Number of S1 entities to dry-run")
    args = parser.parse_args()
    
    run_submission(args.chunk_size, args.threshold, args.dry_run)
