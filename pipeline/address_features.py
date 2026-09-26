import re
import pandas as pd

# US/France (5), India (6), Australia/others (4), US+4, UK, Canada
POSTAL_PATTERN = re.compile(r'\b(\d{4,6}|\d{5}-\d{4}|[a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2}|[a-z]\d[a-z]\s*\d[a-z]\d)\b')
HOUSE_NUM_PATTERN = re.compile(r'(?:^|\s|#|no\.?\s*|d/|h\.?no\.?\s*)0*(\d+)(?:[/-]\d+)*\b')

def extract_postal_code(text: str) -> str:
    if not text: return ""
    # Find all matches, take the last one as postal codes are typically at the end of addresses
    matches = POSTAL_PATTERN.findall(text)
    if matches:
        val = matches[-1]
        if val.isdigit():
            val = val.lstrip('0')
            if not val: val = '0'
        return val
    return ""

def extract_house_number(text: str) -> str:
    if not text: return ""
    match = HOUSE_NUM_PATTERN.search(text)
    if match:
        val = match.group(1).lstrip('0')
        return val if val else '0'
    return ""

def extract_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts structured fields (postal code, house number) and missingness indicators.
    Country is kept as an open-set categorical string variable for unseen test countries like France.
    """
    df = df.copy()
    
    addr_clean = df['business_address_clean']
    
    df['postal_code'] = addr_clean.apply(extract_postal_code)
    df['house_number'] = addr_clean.apply(extract_house_number)
    
    df['name_missing'] = (df['business_name_clean'] == "").astype(int)
    df['address_missing'] = (addr_clean == "").astype(int)
    df['country_missing'] = df['country'].isna().astype(int)
    df['postal_missing'] = (df['postal_code'] == "").astype(int)
    df['house_number_missing'] = (df['house_number'] == "").astype(int)
    
    return df
