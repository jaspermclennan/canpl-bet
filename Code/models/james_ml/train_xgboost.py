import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import joblib
from pathlib import Path
import os

# --- PATH SETUP ---
cwd = Path(os.getcwd())
# Ensure we are at the repo root
REPO_ROOT = cwd if cwd.name == "canpl-bet" else Path(__file__).resolve().parent.parent.parent.parent

DATA_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("\n--- 🤖 TRAINING XGBOOST MODEL ---")
    
    if not DATA_FILE.exists():
        print(f"❌ Missing {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Feature Selection
    features = [
        'diff_total',    # ELO Gap
        'diff_form_pts', # Form Gap (Points)
        'diff_form_gd'   # Form Gap (Goal Diff)
    ]
    target = 'label'
    
    print(f"   Features: {features}")
    
    # 2. Time Split (Train on 2019-2024, Test on 2025)
    # We use 2025 matches as the 'test' to see if the model can predict the future
    split_date = '2025-01-01'
    
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    # Filter out rows where label might be missing (future games)
    train_df = train_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"   Training set: {len(X_train)} matches (Pre-2025)")
    print(f"   Testing set:  {len(X_test)} matches (2025)")
    
    # 3. Define Model
    model = xgb.XGBClassifier(
        n_estimators=100,     # Fixed typo (was n_estimator)
        max_depth=3,          # Keep simple to prevent overfitting
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )
    
    # 4. Train
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test, probs)
    
    print("\n" + "="*30)
    print(f"   RESULTS (2025 Season)")
    print(f"   XGBoost Accuracy: {acc:.1%}")
    print(f"   XGBoost Log Loss: {loss:.4f}")
    print("="*30 + "\n")
    
    # Feature Importance
    print("   What matters most?")
    importance = model.feature_importances_
    for i, f_name in enumerate(features):
        print(f"      {f_name:<15}: {importance[i]:.4f}")
        
    # 6. Save Final Model (Train on EVERYTHING for production)
    print("\n   🚀 Retraining on FULL history for 2026 predictions...")
    full_X = df[features]
    full_y = df[target]
    model.fit(full_X, full_y)
    
    out_path = MODEL_DIR / "xgboost_model.pkl"
    joblib.dump(model, out_path)
    print(f"   ✅ Model saved to {out_path}")

if __name__ == "__main__":
    main()