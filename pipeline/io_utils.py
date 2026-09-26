import pandas as pd
from pathlib import Path
from . import config

def read_tsv_chunked(file_path: Path, chunk_size: int = None):
    """Yields pandas DataFrames in chunks from a TSV file."""
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    
    # We explicitly use sep="\t". 
    # dtype=str is used to avoid pandas inferring numeric types for IDs or ZIP codes.
    return pd.read_csv(
        file_path,
        sep=config.TSV_SEP,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        na_values=['', 'NaN', 'null']
    )

def read_tsv_full(file_path: Path):
    """Read entire TSV. Memory intensive for Source 2 and 3."""
    return pd.read_csv(
        file_path,
        sep=config.TSV_SEP,
        dtype=str,
        keep_default_na=False,
        na_values=['', 'NaN', 'null']
    )

def write_tsv(df: pd.DataFrame, file_path: Path, append=False):
    """Writes a DataFrame to TSV."""
    mode = 'a' if append else 'w'
    header = not append or not file_path.exists()
    df.to_csv(file_path, sep=config.TSV_SEP, index=False, mode=mode, header=header)

def get_source_paths(is_train: bool = True):
    if is_train:
        return {
            'source1': config.TRAIN_SOURCE1,
            'source2': config.TRAIN_SOURCE2,
            'source3': config.TRAIN_SOURCE3,
            'ground_truth': config.TRAIN_GROUND_TRUTH
        }
    else:
        return {
            'source1': config.TEST_SOURCE1,
            'source2': config.TEST_SOURCE2,
            'source3': config.TEST_SOURCE3
        }
