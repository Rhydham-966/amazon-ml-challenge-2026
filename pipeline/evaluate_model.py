import pandas as pd
import numpy as np
import logging
import random
import argparse
import time
import os
import psutil
import gc
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq

from . import config, split, ground_truth, blocking, normalize, address_features
from .feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def f05_score(true_matches: set, pred_matches: set):
    if not true_matches and not pred_matches:
        return 1.0 
    if not true_matches or not pred_matches:
        return 0.0
    
    tp = len(true_matches.intersection(pred_matches))
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)
    
    if tp == 0: return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_evaluation(chunk_size: int, smoke_test: bool):
    logger.info("Initializing Phase 11-13: Validation & Threshold Sweeping (Memory Safe)")
    
    model_path = config.MODELS_DIR / "lgbm_model.txt"
    booster = lgb.Booster(model_file=str(model_path))
    logger.info("Loaded trained LightGBM model.")
    
    val_ids = list(split.get_val_s1_ids())
    random.seed(42)
    sample_ids = set(random.sample(val_ids, min(25000, len(val_ids))))
    
    truth_map = ground_truth.load_ground_truth()
    
    logger.info("Loading S1 Validation Sample...")
    s1_df = pd.read_csv(config.TRAIN_SOURCE1, sep='\t', dtype=str).fillna("")
    s1_df = s1_df[s1_df['entity_id'].isin(sample_ids)]
    s1_df = normalize.normalize_dataframe(s1_df)
    s1_df = address_features.extract_address_features(s1_df)
    
    logger.info("Loading S2/S3 Data...")
    s2_df = pd.read_csv(config.TRAIN_SOURCE2, sep='\t', dtype=str).fillna("")
    s3_df = pd.read_csv(config.TRAIN_SOURCE3, sep='\t', dtype=str).fillna("")
    s_cands = pd.concat([s2_df, s3_df], ignore_index=True)
    s_cands = normalize.normalize_dataframe(s_cands)
    s_cands = address_features.extract_address_features(s_cands)
    
    blocker = blocking.Blocker()
    blocker.load_indexes()
    
    logger.info("Generating candidates via Optimized Blocking...")
    
    output_dir = config.DATA_DIR / "validation_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "val_features_preds.parquet"
    if output_file.exists():
        output_file.unlink()
        
    writer = None
    
    chunk_list = []
    total_candidates = 0
    chunk_idx = 0
    start_time = time.time()
    
    def process_chunk_data(c_list):
        nonlocal writer, total_candidates, chunk_idx
        chunk_idx += 1
        
        df_chunk = pd.DataFrame(c_list)
        
        # build_features extracts text data using merge on these chunks
        features_chunk = build_features(df_chunk, s1_df, s_cands)
        drop_cols = ['s1_id', 's2_id', 'label']
        X = features_chunk.drop(columns=drop_cols, errors='ignore')
        
        probs = booster.predict(X)
        features_chunk['prob'] = np.array(probs, dtype=np.float32)
        
        table = pa.Table.from_pandas(features_chunk)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        
        elapsed = time.time() - start_time
        mem = get_memory_usage()
        logger.info(f"Chunk {chunk_idx} | Cands processed: {total_candidates} | "
                    f"Time: {elapsed:.1f}s | Peak Mem: {mem:.1f} MB | "
                    f"Features: {X.shape[1]} | Out Rows: {len(features_chunk)}")
        
        if smoke_test and chunk_idx == 1:
            logger.info("=== SMOKE TEST RESULTS ===")
            logger.info(f"Feature count: {X.shape[1]}")
            logger.info(f"Feature names: {list(X.columns)}")
            logger.info(f"Data types:\n{X.dtypes}")
            logger.info(f"Missing values:\n{X.isna().sum()}")
            logger.info(f"Output rows: {len(features_chunk)}")
            logger.info(f"Peak memory: {mem:.1f} MB")
            logger.info("==========================")
            
        del df_chunk
        del features_chunk
        del X
        gc.collect()
        
    # Candidate generator logic
    # Do not create python dict containing 113M candidates
    for _, row in tqdm(s1_df.iterrows(), total=len(s1_df), desc="Blocking and Feature Gen"):
        cands = blocker.get_candidates(
            row['country'], row['business_name_clean'], 
            row['business_name_without_legal_suffix'], 
            row['postal_code'], row['house_number'], 
            row['business_address_clean']
        )
        for c in cands:
            chunk_list.append({'s1_id': row['entity_id'], 's2_id': c})
            total_candidates += 1
            
            if len(chunk_list) >= chunk_size:
                process_chunk_data(chunk_list)
                chunk_list = []
                if smoke_test:
                    break
        if smoke_test and len(chunk_list) == 0:
            break
            
    if chunk_list and not smoke_test:
        process_chunk_data(chunk_list)
        
    if writer:
        writer.close()
        
    if smoke_test:
        logger.info("Smoke test completed successfully. Ready for full evaluation.")
        return
        
    logger.info("Streaming predictions for threshold sweep...")
    # Read the parquet file in batches to compute the F0.5
    preds_dict = {}
    parquet_file = pq.ParquetFile(output_file)
    for batch in parquet_file.iter_batches(batch_size=2_500_000, columns=['s1_id', 's2_id', 'prob']):
        batch_df = batch.to_pandas()
        # Filter early to save memory
        batch_df = batch_df[batch_df['prob'] >= 0.1]
        for _, r in batch_df.iterrows():
            preds_dict.setdefault(r['s1_id'], []).append((r['s2_id'], r['prob']))
            
    logger.info("Sweeping Thresholds (0.50 -> 0.95) for Macro F0.5...")
    thresholds = np.arange(0.50, 0.96, 0.05)
    best_t = 0.50
    best_f05 = 0.0
    
    for t in thresholds:
        f05_scores = []
        for eid in sample_ids:
            t_matches = truth_map.get(eid, set())
            cands = preds_dict.get(eid, [])
            p_matches = {c[0] for c in cands if c[1] >= t}
            f05_scores.append(f05_score(t_matches, p_matches))
            
        macro_f05 = np.mean(f05_scores)
        logger.info(f"Threshold {t:.2f} -> Macro F0.5: {macro_f05:.4f}")
        
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_t = t
            
    logger.info("=========================================")
    logger.info(f"Best Validation Threshold: {best_t:.2f}")
    logger.info(f"Best Validation Macro F0.5: {best_f05:.4f}")
    logger.info("=========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=250000, help="Number of candidate pairs per chunk")
    parser.add_argument("--smoke-test", action="store_true", help="Run a small smoke test first")
    args = parser.parse_args()
    
    run_evaluation(chunk_size=args.chunk_size, smoke_test=args.smoke_test)
