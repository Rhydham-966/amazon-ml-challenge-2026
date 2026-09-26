import re
import pandas as pd

# Common abbreviation dictionary matching our prior analysis
ABBR_MAP = {
    'inc': 'incorporated',
    'corp': 'corporation',
    'pvt': 'private',
    'ltd': 'limited',
    'llc': 'limited liability company',
    'llp': 'limited liability partnership',
    'co': 'company',
    'pc': 'professional corporation',
    'lp': 'limited partnership',
    'pllc': 'professional limited liability company',
    'st': 'street',
    'rd': 'road',
    'dr': 'drive',
    'ave': 'avenue',
    'ln': 'lane',
    'ct': 'court',
    'nr': 'near'
}
ABBR_PATTERN = re.compile(r'\b(' + '|'.join(ABBR_MAP.keys()) + r')\b')

# Set of standard legal suffixes to safely strip for blocking
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'pvt', 'private', 'ltd', 'limited',
    'llc', 'llp', 'co', 'company', 'pc', 'lp', 'pllc'
}
SUFFIX_PATTERN = re.compile(r'\b(' + '|'.join(LEGAL_SUFFIXES) + r')\b')


def normalize_text(text: str) -> str:
    """Safe, deterministic text normalization."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Punctuation to space
    text = re.sub(r'[,\\.-]', ' ', text)
    # Fast dictionary lookup
    text = ABBR_PATTERN.sub(lambda x: ABBR_MAP[x.group()], text)
    # Whitespace trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_legal_suffix(clean_name: str) -> str:
    """Removes trailing or internal legal suffixes while preserving core business tokens."""
    if not clean_name:
        return ""
    
    prev = ""
    curr = clean_name
    # Keep stripping until stable (e.g. 'abc private limited' -> 'abc')
    while curr != prev:
        prev = curr
        curr = SUFFIX_PATTERN.sub('', curr)
        curr = re.sub(r'\s+', ' ', curr).strip()
    return curr

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies normalization without overwriting the original raw columns.
    Creates: business_name_clean, business_address_clean, business_name_without_legal_suffix.
    """
    df = df.copy()
    
    df['business_name_clean'] = df['business_name'].apply(normalize_text)
    df['business_address_clean'] = df['business_address'].apply(normalize_text)
    
    df['business_name_without_legal_suffix'] = df['business_name_clean'].apply(remove_legal_suffix)
    
    return df
