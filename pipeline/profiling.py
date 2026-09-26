import logging
from pathlib import Path
from tqdm import tqdm
from . import io_utils
from . import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def profile_source(file_path: Path, name: str):
    """Scalable profiling of a large TSV source file without blowing up memory."""
    logger.info(f"Starting profiling for {name} ({file_path.name})")
    
    total_rows = 0
    missing_names = 0
    missing_addresses = 0
    missing_countries = 0
    unique_names = set()
    total_name_len = 0
    
    # Process in chunks to respect scalability requirements
    try:
        for chunk in tqdm(io_utils.read_tsv_chunked(file_path), desc=f"Profiling {name}"):
            total_rows += len(chunk)
            
            missing_names += chunk['business_name'].isna().sum()
            missing_addresses += chunk['business_address'].isna().sum()
            missing_countries += chunk['country'].isna().sum()
            
            # Aggregate unique names across all chunks efficiently
            names = chunk['business_name'].dropna()
            unique_names.update(names.unique())
            
            total_name_len += names.str.len().sum()
    except Exception as e:
        logger.error(f"Error profiling {name}: {str(e)}")
        raise
        
    avg_name_len = total_name_len / max(1, (total_rows - missing_names))
    
    logger.info(f"--- Profiling Report for {name} ---")
    logger.info(f"Total Rows: {total_rows:,}")
    logger.info(f"Missing Names: {missing_names:,}")
    logger.info(f"Missing Addresses: {missing_addresses:,}")
    logger.info(f"Missing Countries: {missing_countries:,}")
    logger.info(f"Unique Names: {len(unique_names):,}")
    logger.info(f"Average Name Length: {avg_name_len:.1f} chars")
    
    return {
        'total_rows': total_rows,
        'missing_names': missing_names,
        'missing_addresses': missing_addresses,
        'missing_countries': missing_countries,
        'unique_names': len(unique_names),
        'avg_name_len': avg_name_len
    }

def run_profiling():
    """Main entrypoint for the profiling stage."""
    paths = io_utils.get_source_paths(is_train=True)
    
    profile_source(paths['source1'], "Train Source 1")
    profile_source(paths['source2'], "Train Source 2")
    profile_source(paths['source3'], "Train Source 3")
    
    logger.info("Profiling stage completed.")

if __name__ == '__main__':
    run_profiling()
