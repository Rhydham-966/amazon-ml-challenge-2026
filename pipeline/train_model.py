import pandas as pd
import lightgbm as lgb
import logging
from . import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train():
    logger.info("Initializing Phase 10: Primary Model Training")
    
    logger.info("Loading training feature matrix...")
    df = pd.read_parquet(config.DATA_DIR / "train_features.parquet")
    
    drop_cols = ['s1_id', 's2_id', 'label']
    features = [c for c in df.columns if c not in drop_cols]
    
    logger.info(f"Training on {len(df):,} pairs with {len(features)} features.")
    
    X = df[features]
    y = df['label']
    
    logger.info("Training LightGBM model (this is highly optimized and will be fast)...")
    
    # We use tree parameters suitable for a highly unbalanced binary classification task
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X, y)
    
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.MODELS_DIR / "lgbm_model.txt"
    clf.booster_.save_model(out_path)
    
    logger.info("=========================================")
    logger.info(f"Model successfully saved to: {out_path}")
    
    # Print feature importance so we can see what the AI learned!
    importance = pd.DataFrame({
        'feature': features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n--- Top 10 Most Important Features ---")
    for _, r in importance.head(10).iterrows():
        logger.info(f"{r['feature']}: {r['importance']}")
    
    logger.info("Phase 10 Complete!")

if __name__ == "__main__":
    train()
