import logging
from . import split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_split_leakage():
    """Ensures absolute separation between Train and Validation Source 1 Entities."""
    logger.info("Verifying Train/Validation separation...")
    train_ids = split.get_train_s1_ids()
    val_ids = split.get_val_s1_ids()
    
    intersection = train_ids.intersection(val_ids)
    
    if len(intersection) > 0:
        logger.error(f"LEAKAGE DETECTED: {len(intersection)} entities found in both train and validation splits!")
        return False
        
    logger.info("LEAKAGE CHECK: PASS - Train and Validation Source 1 sets are strictly mutually exclusive.")
    return True

if __name__ == "__main__":
    verify_split_leakage()
