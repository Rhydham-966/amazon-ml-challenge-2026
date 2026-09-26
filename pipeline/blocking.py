import pandas as pd
import logging
from typing import Set
from . import config

logger = logging.getLogger(__name__)

STOPWORDS = {'the', 'and', 'of', 'in', 'to', 'for', 'on', 'at', 'by', 'a', 'an'}

def get_tokens(text: str):
    if not text: return []
    return [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]

class Blocker:
    def __init__(self):
        self.indexes = {}
        
    def load_indexes(self):
        logger.info("Loading Advanced inverted indexes into memory...")
        
        index_names = ["name_idx", "name_no_suffix_idx", "postal_idx", "name_token_idx", "address_token_idx"]
        
        for name in index_names:
            path = config.INDEX_DIR / f"{name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                self.indexes[name] = df.set_index('blocking_key')['candidate_ids'].to_dict()
                logger.info(f"Loaded {name}: {len(self.indexes[name]):,} keys.")
        
        logger.info("All Inverted Indexes successfully loaded.")
        
    def _add_candidates(self, candidates: Set[str], index_name: str, key: str, cap: int = None):
        if index_name in self.indexes and key in self.indexes[index_name]:
            cands = self.indexes[index_name][key]
            # The cap naturally ignores massive uninformative buckets like 'US||store' without needing pre-computed IDF
            if cap is None or len(cands) <= cap:
                candidates.update(cands)

    def get_candidates(self, country: str, name_clean: str, name_no_suf: str, postal: str, house: str, addr_clean: str) -> Set[str]:
        candidates = set()
        
        # Rule 1 & 2: Exact Name (No cap, if 50,000 McDonalds exist, we need to check them)
        if name_clean:
            self._add_candidates(candidates, "name_idx", f"{country}||{name_clean}")
        if name_no_suf:
            self._add_candidates(candidates, "name_no_suffix_idx", f"{country}||{name_no_suf}")
                
        # Rule 3: Postal Code (Lowered to 2000 because of poor ROI on huge candidate sets)
        if postal:
            self._add_candidates(candidates, "postal_idx", f"{country}||{postal}", cap=2000)
            
        # Rule 4: Name Tokens (Lowered from 30k to 10k to slice 4B candidates while losing minimal recall)
        if name_no_suf:
            for token in get_tokens(name_no_suf):
                self._add_candidates(candidates, "name_token_idx", f"{country}||{token}", cap=10000)
                
        # Rule 5: House Number + Address Token (Highly specific, bumped to 3000)
        if house and addr_clean:
            for token in get_tokens(addr_clean):
                if token != house:
                    self._add_candidates(candidates, "address_token_idx", f"{country}||{house}||{token}", cap=3000)
                    
        return candidates
