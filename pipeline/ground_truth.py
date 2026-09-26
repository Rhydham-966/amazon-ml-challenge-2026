import pandas as pd
from typing import Dict, Set
from . import io_utils
from . import config

def load_ground_truth() -> Dict[str, Set[str]]:
    """
    Parses the ground truth TSV into a dictionary mapping:
    Source1 ID -> Set of True Matching IDs (from Source 2 / Source 3)
    Empty matches (singletons) are correctly parsed as empty sets set().
    """
    df = io_utils.read_tsv_full(config.TRAIN_GROUND_TRUTH)
    
    truth_map = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        
        if pd.isna(matches_str) or str(matches_str).strip() == "":
            truth_map[s1_id] = set()
        else:
            truth_map[s1_id] = set(str(matches_str).split(','))
            
    return truth_map

def get_true_matches(truth_map: Dict[str, Set[str]], s1_id: str) -> Set[str]:
    return truth_map.get(s1_id, set())

def is_match(truth_map: Dict[str, Set[str]], s1_id: str, candidate_id: str) -> bool:
    return candidate_id in get_true_matches(truth_map, s1_id)
