import os
from pathlib import Path

# --- DIRECTORY STRUCTURE ---
ROOT_DIR = Path("E:/ML/amazon/student_resource")
DATA_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

PIPELINE_DIR = ROOT_DIR / "pipeline"
ARTIFACT_DIR = ROOT_DIR / "pipeline_artifacts"
OUTPUT_DIR = ROOT_DIR / "output"

# Ensure essential directories exist
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- INPUT FILES (TRAIN) ---
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# --- INPUT FILES (TEST) ---
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# --- ARTIFACT PATHS ---
SPLIT_DIR = ARTIFACT_DIR / "splits"
INDEX_DIR = ARTIFACT_DIR / "indexes"
TFIDF_DIR = ARTIFACT_DIR / "tfidf"
MODELS_DIR = ARTIFACT_DIR / "models"
EXPERIMENTS_DIR = ARTIFACT_DIR / "experiments"

# Create artifact subdirectories
for d in [SPLIT_DIR, INDEX_DIR, TFIDF_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- CHUNKING & SCALING ---
CHUNK_SIZE = 100000  # Number of rows to read into memory per chunk

# --- VALIDATION ---
RANDOM_SEED = 42
TRAIN_RATIO = 0.8  # 80% train, 20% validation at S1 entity level

# --- OUTPUT FILES ---
MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"

# --- SCHEMA / TYPES ---
TSV_SEP = "\t"
EXPECTED_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']
