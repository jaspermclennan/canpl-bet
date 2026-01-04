import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import os

# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent.parent

DATA_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("--- TRAINING FINAL PRODUCTION MODEL ---")
    
    if not DATA_FILE.exists():
        print(f" Missing {DATA_FILE}")
        return
    
    # 1. Load Everything
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    features = ['diff_total', 'diff_form_pts', 'diff_form_gd']
    target = 'label'
    
    # Filter out future placeholder games (rows with no result yet)
    train_df = df.dropna(subset=[target])
    
    X = train_df[features]
    y = train_df[target]
    
    print(f"   Training on ALL available history: {len(X)} matches")
    
    # 2. Configure the "Winner" Settings
    # These are the values your tuning script found.
    # We hardcode them here so this script is always "Production Ready".
    final_params = {
        'max_depth': 2,               # Kept it simple (Best found)
        'learning_rate': 0.01,        # Slow and steady learning
        'n_estimators': 100,          # Number of trees
        'subsample': 1.0,             # Use all rows
        'colsample_bytree': 0.8,      # Use 80% of features per tree
        'gamma': 5,                   # Conservative splitting
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }
    
    # 3. Train the Model
    # We removed 'use_label_encoder' to fix your warning spam
    model = xgb.XGBClassifier(**final_params)
    model.fit(X, y)
    
    # 4. Save
    out_path = MODEL_DIR / "xgboost_model.pkl"
    joblib.dump(model, out_path)
    
    print("-" * 30)
    print(f"  Model saved to: {out_path}")
    print("   This model is now trained on 2019-2025 data.")
    print("   It is ready to predict 2026 games.")
    print("-" * 30)
    
    # 5. Feature Importance Check
    print("   Final Feature Weights:")
    importance = model.feature_importances_
    for i, f_name in enumerate(features):
        print(f"      {f_name:<15}: {importance[i]:.4f}")

if __name__ == "__main__":
    main()