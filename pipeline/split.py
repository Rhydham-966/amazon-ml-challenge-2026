import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from . import config
from . import io_utils

logger = logging.getLogger(__name__)

def generate_splits():
    """Generates the Source 1 Train/Val split and persists the IDs to prevent leakage."""
    val_ids_path = config.SPLIT_DIR / "validation_source1_ids.json"
    train_ids_path = config.SPLIT_DIR / "train_source1_ids.json"
    
    if val_ids_path.exists() and train_ids_path.exists():
        logger.info("Splits already exist. Reusing them to ensure deterministic reproducibility.")
        return
        
    logger.info("Reading Source 1 IDs to generate train/validation split...")
    s1_df = io_utils.read_tsv_full(config.TRAIN_SOURCE1)
    all_s1_ids = s1_df['entity_id'].unique()
    
    train_ids, val_ids = train_test_split(
        all_s1_ids, 
        train_size=config.TRAIN_RATIO, 
        random_state=config.RANDOM_SEED
    )
    
    # Save as lists
    with open(train_ids_path, 'w') as f:
        json.dump(list(train_ids), f)
        
    with open(val_ids_path, 'w') as f:
        json.dump(list(val_ids), f)
        
    logger.info(f"Split completed. Training S1 entities: {len(train_ids):,}. Validation S1 entities: {len(val_ids):,}")

def get_train_s1_ids() -> set:
    with open(config.SPLIT_DIR / "train_source1_ids.json", 'r') as f:
        return set(json.load(f))

def get_val_s1_ids() -> set:
    with open(config.SPLIT_DIR / "validation_source1_ids.json", 'r') as f:
        return set(json.load(f))

if __name__ == "__main__":
    generate_splits()
